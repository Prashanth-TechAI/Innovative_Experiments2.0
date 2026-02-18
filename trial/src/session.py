from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference
from bson import ObjectId, errors as bson_errors
import requests
import logging
from typing import Optional

logger = logging.getLogger("mcp.session")
logger.setLevel(logging.DEBUG)

class Session:
    def __init__(self, config):

        self.config = config
        self.mongo: Optional[MongoClient] = None
        self.atlas: Optional[requests.Session] = None
        self._company_id: Optional[ObjectId] = None

    @property
    def current_company_id(self) -> Optional[ObjectId]:
        return self._company_id

    @current_company_id.setter
    def current_company_id(self, cid: str):
        try:
            self._company_id = ObjectId(cid)
            logger.info("Using company_id %s", self._company_id)
        except (bson_errors.InvalidId, TypeError) as e:
            logger.error("Invalid company_id '%s': %s", cid, e)
            raise ValueError(f"Invalid company_id: {cid}") from e
        except Exception as e:
            logger.exception("Unexpected error setting company_id '%s'", cid)
            raise

    def connect(self):
        logger.debug(
            "About to connect to MongoDB at %s with read_preference=%s",
            self.config.mongo_uri,
            self.config.read_preference
        )
        try:
            rp = (
                ReadPreference.SECONDARY_PREFERRED
                if self.config.read_preference.lower() == "secondarypreferred"
                else ReadPreference.PRIMARY
            )
            self.mongo = MongoClient(
                self.config.mongo_uri,
                appname="MCP-Python",
                read_preference=rp
            )
        except Exception as e:
            logger.exception("Failed to instantiate MongoClient")
            raise RuntimeError("Could not connect to MongoDB") from e

        try:
            info = self.mongo.server_info()
            logger.info(
                "Connected to MongoDB at %s (readPreference=%s); server version: %s",
                self.config.mongo_uri,
                rp.mode,
                info.get("version", "unknown")
            )
        except Exception as e:
            logger.exception("Connected but failed to get server_info")
            raise RuntimeError("MongoDB connection established but server_info failed") from e

        if getattr(self.config, "api_client_id", None) and getattr(self.config, "api_client_secret", None):
            try:
                self.atlas = requests.Session()
                self.atlas.auth = (
                    self.config.api_client_id,
                    self.config.api_client_secret
                )
                self.atlas.headers.update({"Accept": "application/json"})
                logger.info("Initialized Atlas REST client for %s", self.config.api_base_url)
            except Exception as e:
                logger.exception("Failed to initialize Atlas REST client; continuing without it")

    def close(self):
        if not self.mongo:
            return
        try:
            self.mongo.close()
            logger.info("Closed MongoDB connection")
        except Exception as e:
            logger.exception("Error closing MongoDB connection")

    def ensure_connected(self):
        if not self.mongo:
            logger.error("MongoDB client not connected.")
            raise RuntimeError("MongoDB client not connected. Call connect() first.")

    def get_db(self, company_id: Optional[ObjectId] = None):
        try:
            self.ensure_connected()
            if self._company_id is None:
                logger.error("No company_id set on session.")
                raise RuntimeError("No company_id set on session.")
            return self.mongo[self.config.db_name]
        except Exception:
            logger.exception("Error getting database '%s'", self.config.db_name)
            raise

    def get_collection(self, coll_name: str, company_id: Optional[ObjectId] = None):
        try:
            db = self.get_db(company_id)
            allowed = getattr(self.config, "allowed_collections", None)
            if allowed is not None and coll_name not in allowed:
                logger.warning("Access to collection '%s' is not allowed", coll_name)
                raise PermissionError(f"Collection '{coll_name}' is not in allowed_collections.")
            return db[coll_name]
        except PermissionError:
            raise
        except Exception:
            logger.exception("Error getting collection '%s'", coll_name)
            raise RuntimeError(f"Could not get collection '{coll_name}'") from None
