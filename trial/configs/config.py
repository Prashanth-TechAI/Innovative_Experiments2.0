import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Literal

@dataclass
class DisabledTools:
    """Control which tools are not registered."""
    categories: List[str] = field(default_factory=list)
    names:      List[str] = field(default_factory=list)
    types:      List[str] = field(default_factory=list)

@dataclass
class Config:
    company_id: str
    mongo_uri: str
    db_name: str
    read_preference: str
    allowed_collections: Optional[List[str]]
    non_tenant_collections: List[str]
    log_path: str
    log_level: str
    telemetry: Literal["enabled", "disabled"]
    read_only: bool
    disabled_tools: DisabledTools
    api_base_url: str
    api_client_id: str
    api_client_secret: str
    openai_api_key: Optional[str]
    model_name: Optional[str]

def load_config() -> Config:
    parser = argparse.ArgumentParser(description="MongoDB MCP Server Configuration")
    parser.add_argument(
        "--companyId",
        default=os.getenv("COMPANY_ID", ""),
        help="Your tenant’s company ID (ObjectId)."
    )
    parser.add_argument(
        "--mongoUri",
        default=os.getenv("MONGO_URI", "mongodb+srv://prashanth_01:prashanth123@cluster0.pkiva.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"),
        help="MongoDB connection URI"
    )
    parser.add_argument(
        "--dbName",
        default=os.getenv("DB_NAME", "test"),
        help="MongoDB database name"
    )
    parser.add_argument(
        "--readPreference",
        default=os.getenv("MDB_MCP_READ_PREF", "secondaryPreferred"),
        help="MongoDB readPreference (default: secondaryPreferred)"
    )
    parser.add_argument(
        "--collections",
        default=os.getenv("COLLECTIONS", "*"),
        help="Comma-separated allowed collections, or '*' for all"
    )
    parser.add_argument(
        "--nonTenantCollections",
        default=os.getenv("NON_TENANT_COLLECTIONS", "plans,countries,states,cities"),
        help="Comma-separated list of collections to skip tenant filtering"
    )
    parser.add_argument(
        "--logPath",
        default=os.getenv(
            "LOG_PATH",
            os.path.expanduser("~/.mongodb/mongodb-mcp/.app-logs/mcp.log")
        ),
        help="Path to the rotating log file"
    )
    parser.add_argument(
        "--logLevel",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARN, ERROR)"
    )
    parser.add_argument(
        "--readOnly",
        action="store_true",
        default=os.getenv("MDB_MCP_READ_ONLY", "false").lower() in ("1","true","yes"),
        help="If set, disables all write/update/delete tools"
    )
    parser.add_argument(
        "--disableToolCategories",
        default=os.getenv("MDB_MCP_DISABLED_TOOL_CATEGORIES", ""),
        help="Comma-separated MCP tool categories to disable"
    )
    parser.add_argument(
        "--disableToolNames",
        default=os.getenv("MDB_MCP_DISABLED_TOOL_NAMES", ""),
        help="Comma-separated MCP tool names to disable"
    )
    parser.add_argument(
        "--disableToolTypes",
        default=os.getenv("MDB_MCP_DISABLED_TOOL_TYPES", ""),
        help="Comma-separated MCP tool operation types to disable (e.g. create, update)"
    )
    parser.add_argument(
        "--telemetry",
        choices=["enabled", "disabled"],
        default=(
            os.getenv("DO_NOT_TRACK", "") == "1" and "disabled"
        ) or os.getenv("MDB_MCP_TELEMETRY", "enabled"),
        help="Enable or disable telemetry collection"
    )
    parser.add_argument(
        "--apiBaseUrl",
        default=os.getenv("API_BASE_URL", "https://cloud.mongodb.com/"),
        help="Base URL for the Atlas API"
    )
    parser.add_argument(
        "--apiClientId",
        default=os.getenv("API_CLIENT_ID", ""),
        help="Atlas API public key"
    )
    parser.add_argument(
        "--apiClientSecret",
        default=os.getenv("API_CLIENT_SECRET", ""),
        help="Atlas API private key"
    )
    parser.add_argument(
        "--openaiApiKey",
        default=os.getenv("OPENAI_API_KEY", None),
        help="OpenAI API key for function-calling (optional)"
    )
    parser.add_argument(
        "--modelName",
        default=os.getenv("MODEL_NAME", None),
        help="OpenAI model name (e.g. gpt-4o-mini)"
    )
    args, _ = parser.parse_known_args()
    raw_colls = args.collections.strip()
    allowed = None if raw_colls in ("*", "") else [c.strip() for c in raw_colls.split(",") if c.strip()]
    non_tenant = [c.strip() for c in args.nonTenantCollections.split(",") if c.strip()]

    def parse_list(val: str) -> List[str]:
        return [] if not val.strip() else [x.strip() for x in val.split(",") if x.strip()]

    disabled = DisabledTools(
        categories=parse_list(args.disableToolCategories),
        names=parse_list(args.disableToolNames),
        types=parse_list(args.disableToolTypes),
    )

    return Config(
        company_id=args.companyId,
        mongo_uri=args.mongoUri,
        db_name=args.dbName,
        read_preference=args.readPreference,
        allowed_collections=allowed,
        non_tenant_collections=non_tenant,
        log_path=args.logPath,
        log_level=args.logLevel.upper(),
        read_only=args.readOnly,
        disabled_tools=disabled,
        telemetry=args.telemetry,
        api_base_url=args.apiBaseUrl,
        api_client_id=args.apiClientId,
        api_client_secret=args.apiClientSecret,
        openai_api_key=args.openaiApiKey,
        model_name=args.modelName,
    )
