import logging
from typing import Optional, Dict, Any
from bson import ObjectId, errors as bson_errors
from fastapi import HTTPException, status

logger = logging.getLogger("mcp.company_id_utils")
logger.setLevel(logging.DEBUG)

def make_company_filter(
    collection_name: str,
    company_id: str,
    extra_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    if not isinstance(collection_name, str):
        logger.error("Invalid type for collection_name: %r", collection_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="collection_name must be a string"
        )
    if extra_filter is not None and not isinstance(extra_filter, dict):
        logger.error("Invalid type for extra_filter: %r", extra_filter)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="extra_filter must be a dict"
        )

    try:
        obj_id = ObjectId(company_id)
    except (bson_errors.InvalidId, TypeError) as e:
        logger.exception("Invalid company_id provided: %s", company_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company_id"
        )
    except Exception:
        logger.exception("Unexpected error converting company_id: %s", company_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

    try:
        if collection_name == "companies":
            base = {"_id": obj_id}
        else:
            base = {"company": obj_id}

        if extra_filter:
            base.update(extra_filter)

        return base

    except Exception:
        logger.exception(
            "Error constructing filter for collection '%s', company_id '%s'",
            collection_name, company_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
