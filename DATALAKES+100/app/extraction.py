import asyncio
import logging
from typing import Any, Dict, List

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_fixed

from app.config import (
    COLLECTIONS_TO_EXTRACT,
    DATABASE_NAME,
    MONGO_URI,
)
from app.utils import (
    clean_and_convert_value,
    flatten_dict,
    sanitize_company_name,
)
from app.postgres_utils import async_session, insert_documents_async


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]

REQUIRED_FIELDS = {
    "brokers": [
        "name", "commissionPercent", "yearStartedInRealEstate",
        "realEstateLicenseDetails_licenseNo", "realEstateLicenseDetails_licenseIssueDate",
        "realEstateLicenseDetails_licenseExpirationDate", "licenseStatus",
        "country", "state", "city", "createdAt", "updatedAt"
    ],
    "lands": [
        "name", "propertyType", "purchasePrice", "currentMarketValue", "rentalIncome",
        "plotSize", "plotType", "status", "amenities", "isConstructionExists", "isOnLease",
        "occupancyStatus", "isAgricultural", "country", "state", "city", "createdAt", "updatedAt"
    ],
    "leads": [
        "name", "sourceType", "propertyType", "project", "minBudget", "maxBudget",
        "buyingTimeline", "leadStatus", "rotationCount", "lastActivity", "status",
        "createdAt", "commissionPercent"
    ],
    "properties": [
        "propertyType", "blockName", "floorName", "series", "shopNo", "flatNo", "furnishedStatus",
        "minBudget", "maxBudget", "facing", "vastuCompliant", "carpetArea", "builtUpArea",
        "superBuiltUpArea", "builtUpAreaType", "superBuiltUpAreaType", "noOfBalconies",
        "noOfBathRooms", "noOfBedRooms", "noOfKitchens", "noOfDrawingRooms", "noOfParkingLots",
        "propertyStatus", "status", "bookingDate", "createdAt", "updatedAt"
    ],
    "projects": [
        "name", "slug", "projectType", "projectStatus", "minBudget", "maxBudget", "startDate",
        "completionDate", "land", "noOfPhaseResidential", "noOfUnitsResidential",
        "noOfPhaseCommercial", "noOfUnitsCommercial", "noOfBlocksCommercial",
        "totalRentRevenue", "totalRentAmountRequired", "totalRentedUnits",
        "totalSaleRevenue", "totalSaleAmountRequired", "totalSoldUnits",
        "totalBookedUnits", "totalBookingCancelled", "totalRefundedAmount", "amenities",
        "isGovtApproved", "locationType", "country", "state", "city", "createdAt", "updatedAt"
    ],
    "rent-payments": [
        "project", "property", "tenant", "startDate", "endDate", "amount", "amountPaid",
        "paymentStatus", "paymentMode", "receivedOn", "createdAt", "updatedAt"
    ],
    "tenants": [
        "name", "project", "property", "totalMember", "bookingDate", "bookingType", "rentAmount",
        "rentIncrement", "depositAmount", "depositPaymentMode", "rentStartDate",
        "rentAgreementStartDate", "rentAgreementEndDate", "rentPaymentGeneratedOn",
        "leavingDate", "dueRentAmount", "tenantStatus", "status", "createdAt", "updatedAt",
        "isPet", "isPoliceVerificationDone"
    ]
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def _async_try_lookup(
    collection: str,
    _id: ObjectId,
    field: str = "name",
) -> str | None:
    doc = await db[collection].find_one({"_id": _id}, {field: 1})
    return doc.get(field) if doc else None


async def _async_get_simple_name(
    collection: str,
    value: Any,
    field: str = "name",
) -> str:
    try:
        _id = ObjectId(value) if not isinstance(value, ObjectId) else value
    except Exception:
        return str(value)
    return await _async_try_lookup(collection, _id, field) or str(value)


async def _async_get_property_label(value: Any) -> str:
    try:
        _id = ObjectId(value) if not isinstance(value, ObjectId) else value
    except Exception:
        return str(value)

    doc = await db["properties"].find_one({"_id": _id})
    if not doc:
        return str(value)
    return (
        doc.get("name")
        or " ".join(
            filter(
                None,
                [
                    doc.get("propertyType"),
                    doc.get("blockName"),
                    doc.get("floorName"),
                ],
            )
        )
        or "UnknownProperty"
    )


async def _async_get_state_name(value: Any) -> str:
    try:
        _id = ObjectId(value) if not isinstance(value, ObjectId) else value
    except Exception:
        return str(value)

    doc = await db["countries"].find_one({"states._id": _id}, {"states": 1})
    if doc:
        for st in doc["states"]:
            if st["_id"] == _id:
                return st.get("name", str(value))
    return str(value)


async def _async_get_city_name(value: Any) -> str:
    try:
        _id = ObjectId(value) if not isinstance(value, ObjectId) else value
    except Exception:
        return str(value)

    doc = await db["countries"].find_one(
        {"states.cities._id": _id}, {"states": 1}
    )
    if doc:
        for st in doc["states"]:
            for ct in st.get("cities", []):
                if ct["_id"] == _id:
                    return ct.get("name", str(value))
    return str(value)


async def _async_get_amenities_names(value: Any) -> str:
    async def _single(v):
        return await _async_get_simple_name("amenities", v)

    if isinstance(value, list):
        names = [await _single(x) for x in value]
        return ", ".join(names)
    return await _single(value)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Central async reference map
# ═══════════════════════════════════════════════════════════════════════════════
ASYNC_REF_MAP: Dict[str, tuple[str, str] | str] = {
    # simple 1-to-1 collections
    "company":  ("companies",  "name"),
    "project":  ("projects",   "name"),
    "tenant":   ("tenants",    "name"),
    "land":     ("lands",      "name"),
    "broker":   ("brokers",    "name"),

    # specials (handled by a coroutine)
    "property":  "_async_get_property_label",
    "country":   ("countries", "name"),
    "state":     "_async_get_state_name",
    "city":      "_async_get_city_name",
    "amenities": "_async_get_amenities_names",
}

# map string names → actual callables
_CUSTOM_FUNCS = {
    "_async_get_property_label": _async_get_property_label,
    "_async_get_state_name":     _async_get_state_name,
    "_async_get_city_name":      _async_get_city_name,
    "_async_get_amenities_names": _async_get_amenities_names,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Replacement logic (recursively walks any dict / list)
# ═══════════════════════════════════════════════════════════════════════════════
async def async_replace_ids_with_names(doc: Any) -> Any:
    if isinstance(doc, list):
        return [await async_replace_ids_with_names(x) for x in doc]
    if not isinstance(doc, dict):
        return doc

    result: Dict[str, Any] = {}
    for k, v in doc.items():
        # recurse first
        if isinstance(v, (dict, list)):
            v = await async_replace_ids_with_names(v)

        # look-up if key is configured
        if k in ASYNC_REF_MAP:
            ref_spec = ASYNC_REF_MAP[k]

            # simple collection
            if isinstance(ref_spec, tuple):
                coll, field = ref_spec
                v = await _async_get_simple_name(coll, v, field)

            # custom coroutine
            else:
                v = await _CUSTOM_FUNCS[ref_spec](v)

        result[k] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  (Everything below is your original, approved sync-schema logic)
#     --- unchanged except the call still points to async_replace_ids_with_names
# ═══════════════════════════════════════════════════════════════════════════════

def filter_fields(doc: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    required = REQUIRED_FIELDS.get(collection_name.lower())
    return (
        {k: v for k, v in doc.items() if k in required or k == "original_id"}
        if required
        else doc
    )


async def _async_extract_collection_data(
    collection_name: str, company_id: ObjectId
) -> List[Dict[str, Any]]:
    if collection_name not in COLLECTIONS_TO_EXTRACT:
        return []

    cursor = db[collection_name].find({"company": company_id}, batch_size=1000)
    docs: List[Dict[str, Any]] = []

    async for raw in cursor:
        enriched = await async_replace_ids_with_names(raw)
        clean = {k: clean_and_convert_value(v) for k, v in enriched.items()}
        clean["original_id"] = str(raw["_id"])

        flat = flatten_dict(filter_fields(clean, collection_name))
        flat["Collection"] = collection_name.title()
        docs.append(flat)

    return docs


async def extract_clean_dataset(company_id_str: str):
    company_id = ObjectId(company_id_str)
    company = await db["companies"].find_one({"_id": company_id}, {"name": 1})
    schema_name = sanitize_company_name(company.get("name", "unknown"))

    for coll in COLLECTIONS_TO_EXTRACT:
        try:
            rows = await _async_extract_collection_data(coll, company_id)
            await insert_documents_async(schema_name, coll, rows)
        except Exception as exc:
            logger.error("❌ Sync error %s.%s: %s", schema_name, coll, exc)

    return {
        "company_id": company_id_str,
        "company_name": company.get("name", "unknown"),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Watcher logic, already approved – kept verbatim
# ───────────────────────────────────────────────────────────────────────────────
async def watch_changes(company_id_str: str):
    company_id = ObjectId(company_id_str)
    company = await db["companies"].find_one({"_id": company_id}, {"name": 1})
    schema_name = sanitize_company_name(company.get("name", "unknown"))

    async def _handle_change(change):
        coll = change["ns"]["coll"]
        if coll not in COLLECTIONS_TO_EXTRACT:
            return

        op = change["operationType"]
        doc = None

        if op == "delete":
            _id = change["documentKey"]["_id"]
            async with async_session() as sess:
                await sess.execute(
                    text(
                        f'DELETE FROM "{schema_name}"."{coll}" '
                        'WHERE original_id = :oid'
                    ),
                    {"oid": str(_id)},
                )
                await sess.commit()
            return

        if op in ("insert", "replace"):
            doc = change.get("fullDocument")
        elif op == "update":
            _id = change["documentKey"]["_id"]
            doc = await db[coll].find_one({"_id": _id})

        if not doc or doc.get("company") != company_id:
            return

        enriched = await async_replace_ids_with_names(doc)
        clean = {k: clean_and_convert_value(v) for k, v in enriched.items()}
        clean["original_id"] = str(doc["_id"])

        flat = flatten_dict(filter_fields(clean, coll))
        flat["Collection"] = coll.title()

        await insert_documents_async(schema_name, coll, [flat])

    async def _watch(coll_name: str):
        pipeline = [
            {"$match": {"operationType": {"$in": ["insert", "update", "replace", "delete"]}}}
        ]
        while True:
            try:
                async with db[coll_name].watch(
                    pipeline, full_document="updateLookup"
                ) as stream:
                    async for change in stream:
                        await _handle_change(change)
            except Exception:
                await asyncio.sleep(5)

    for coll in COLLECTIONS_TO_EXTRACT:
        asyncio.create_task(_watch(coll))