"""
Database initialization - creates collections and indexes on startup.
"""
from app.database import get_database
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initialize database collections and indexes.
    This function is idempotent - safe to run multiple times.
    """
    try:
        db = get_database()
        existing_collections = db.list_collection_names()
        
        # 1. UserTable collection
        if "UserTable" not in existing_collections:
            logger.info("Creating UserTable collection...")
            user_table = db["UserTable"]
            user_table.create_index("email", unique=True, name="email_unique")
            logger.info("✓ Created UserTable with unique index on 'email'")
        else:
            logger.debug("UserTable collection already exists. Ensuring indexes...")
            user_table = db["UserTable"]
            try:
                user_table.create_index("email", unique=True, name="email_unique")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 2. AgentConfig collection
        if "AgentConfig" not in existing_collections:
            logger.info("Creating AgentConfig collection...")
            agent_config = db["AgentConfig"]
            # Compound unique indexes: same agent_type/agent_id can exist in different orgs
            agent_config.create_index([("agent_type", 1), ("org_id", 1)], unique=True, name="agent_type_org_unique")
            agent_config.create_index([("agent_id", 1), ("org_id", 1)], unique=True, name="agent_id_org_unique")
            agent_config.create_index("org_id", name="org_id_index")
            logger.info("✓ Created AgentConfig with compound unique indexes on 'agent_type+org_id' and 'agent_id+org_id'")
        else:
            logger.debug("AgentConfig collection already exists. Ensuring indexes...")
            agent_config = db["AgentConfig"]
            try:
                # Drop old global unique indexes if they exist
                try:
                    agent_config.drop_index("agent_type_unique")
                    logger.info("Dropped old agent_type_unique index")
                except Exception:
                    pass  # Index doesn't exist
                try:
                    agent_config.drop_index("agent_id_unique")
                    logger.info("Dropped old agent_id_unique index")
                except Exception:
                    pass  # Index doesn't exist
                
                # Create new compound unique indexes
                agent_config.create_index([("agent_type", 1), ("org_id", 1)], unique=True, name="agent_type_org_unique")
                agent_config.create_index([("agent_id", 1), ("org_id", 1)], unique=True, name="agent_id_org_unique")
                agent_config.create_index("org_id", name="org_id_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 3. Audience collection
        if "Audience" not in existing_collections:
            logger.info("Creating Audience collection...")
            audience = db["Audience"]
            audience.create_index("audience_name", unique=True, name="audience_name_unique")
            audience.create_index("phone_number", name="phone_number_index")
            logger.info("✓ Created Audience with unique index on 'audience_name'")
        else:
            logger.debug("Audience collection already exists. Ensuring indexes...")
            audience = db["Audience"]
            try:
                audience.create_index("audience_name", unique=True, name="audience_name_unique")
                audience.create_index("phone_number", name="phone_number_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 4. Campaigns collection
        if "Campaigns" not in existing_collections:
            logger.info("Creating Campaigns collection...")
            campaigns = db["Campaigns"]
            campaigns.create_index("campaign_name", unique=True, name="campaign_name_unique")
            logger.info("✓ Created Campaigns with unique index on 'campaign_name'")
        else:
            logger.debug("Campaigns collection already exists. Ensuring indexes...")
            campaigns = db["Campaigns"]
            try:
                campaigns.create_index("campaign_name", unique=True, name="campaign_name_unique")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 5. CallLogs collection
        if "CallLogs" not in existing_collections:
            logger.info("Creating CallLogs collection...")
            call_logs = db["CallLogs"]
            call_logs.create_index("meeting_id", unique=True, name="meeting_id_unique")
            logger.info("✓ Created CallLogs with unique index on 'meeting_id'")
        else:
            logger.debug("CallLogs collection already exists. Ensuring indexes...")
            call_logs = db["CallLogs"]
            try:
                call_logs.create_index("meeting_id", unique=True, name="meeting_id_unique")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 6. PhoneNumber collection
        if "PhoneNumber" not in existing_collections:
            logger.info("Creating PhoneNumber collection...")
            phone_number = db["PhoneNumber"]
            phone_number.create_index("phone_number", unique=True, name="phone_number_unique")
            phone_number.create_index("provider", name="provider_index")
            phone_number.create_index("org_id", name="org_id_index")
            phone_number.create_index("agent_type", name="agent_type_index")
            phone_number.create_index([("org_id", 1), ("agent_type", 1)], name="org_agent_compound")
            logger.info("✓ Created PhoneNumber with indexes")
        else:
            logger.debug("PhoneNumber collection already exists. Ensuring indexes...")
            phone_number = db["PhoneNumber"]
            try:
                phone_number.create_index("phone_number", unique=True, name="phone_number_unique")
                phone_number.create_index("provider", name="provider_index")
                phone_number.create_index("org_id", name="org_id_index")
                phone_number.create_index("agent_type", name="agent_type_index")
                phone_number.create_index([("org_id", 1), ("agent_type", 1)], name="org_agent_compound")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 7. Members collection
        if "Members" not in existing_collections:
            logger.info("Creating Members collection...")
            members = db["Members"]
            # Compound unique index: same email can exist in multiple orgs, but not twice in same org
            members.create_index([("email", 1), ("org_id", 1)], unique=True, name="email_org_unique")
            members.create_index("org_id", name="org_id_index")
            members.create_index("email", name="email_index")
            logger.info("✓ Created Members with compound unique index on 'email' + 'org_id'")
        else:
            logger.debug("Members collection already exists. Ensuring indexes...")
            members = db["Members"]
            try:
                members.create_index([("email", 1), ("org_id", 1)], unique=True, name="email_org_unique")
                members.create_index("org_id", name="org_id_index")
                members.create_index("email", name="email_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 8. Integrations collection
        if "Integrations" not in existing_collections:
            logger.info("Creating Integrations collection...")
            integrations = db["Integrations"]
            # Compound unique index: same org_id + model combination must be unique
            integrations.create_index([("org_id", 1), ("model", 1)], unique=True, name="org_model_unique")
            integrations.create_index("org_id", name="org_id_index")
            logger.info("✓ Created Integrations with compound unique index on 'org_id' + 'model'")
        else:
            logger.debug("Integrations collection already exists. Ensuring indexes...")
            integrations = db["Integrations"]
            try:
                integrations.create_index([("org_id", 1), ("model", 1)], unique=True, name="org_model_unique")
                integrations.create_index("org_id", name="org_id_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        # 9. KnowledgeDocuments collection (org-scoped PDF knowledge base)
        if "KnowledgeDocuments" not in existing_collections:
            logger.info("Creating KnowledgeDocuments collection...")
            kd = db["KnowledgeDocuments"]
            kd.create_index("document_id", unique=True, name="document_id_unique")
            kd.create_index("org_id", name="org_id_index")
            logger.info("✓ Created KnowledgeDocuments with indexes on document_id and org_id")
        else:
            logger.debug("KnowledgeDocuments collection already exists. Ensuring indexes...")
            kd = db["KnowledgeDocuments"]
            try:
                kd.create_index("document_id", unique=True, name="document_id_unique")
                kd.create_index("org_id", name="org_id_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")

        # 10. Batches collection (immutable CSV uploads)
        if "Batches" not in existing_collections:
            logger.info("Creating Batches collection...")
            batches = db["Batches"]
            batches.create_index("batch_id", unique=True, name="batch_id_unique")
            batches.create_index("org_id", name="org_id_index")
            batches.create_index(
                [("org_id", 1), ("batch_name", 1)],
                unique=True,
                name="org_batch_name_unique",
                partialFilterExpression={"batch_name": {"$type": "string"}},
            )
            batches.create_index([("org_id", 1), ("agent_type", 1), ("created_at", -1)], name="org_agent_created_at")
            batches.create_index([("execution_status", 1), ("scheduled_at_utc", 1)], name="execution_scheduled_at_index")
            logger.info("✓ Created Batches with indexes")
        else:
            logger.debug("Batches collection already exists. Ensuring indexes...")
            batches = db["Batches"]
            try:
                batches.create_index("batch_id", unique=True, name="batch_id_unique")
                batches.create_index("org_id", name="org_id_index")
                batches.create_index(
                    [("org_id", 1), ("batch_name", 1)],
                    unique=True,
                    name="org_batch_name_unique",
                    partialFilterExpression={"batch_name": {"$type": "string"}},
                )
                batches.create_index([("org_id", 1), ("agent_type", 1), ("created_at", -1)], name="org_agent_created_at")
                batches.create_index([("execution_status", 1), ("scheduled_at_utc", 1)], name="execution_scheduled_at_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")

        # 11. BatchContacts collection (parsed per-contact rows)
        if "BatchContacts" not in existing_collections:
            logger.info("Creating BatchContacts collection...")
            batch_contacts = db["BatchContacts"]
            batch_contacts.create_index([("batch_id", 1), ("row_number", 1)], unique=True, name="batch_row_unique")
            batch_contacts.create_index([("org_id", 1), ("agent_type", 1)], name="org_agent_index")
            batch_contacts.create_index([("batch_id", 1), ("status", 1)], name="batch_status_index")
            logger.info("✓ Created BatchContacts with indexes")
        else:
            logger.debug("BatchContacts collection already exists. Ensuring indexes...")
            batch_contacts = db["BatchContacts"]
            try:
                batch_contacts.create_index([("batch_id", 1), ("row_number", 1)], unique=True, name="batch_row_unique")
                batch_contacts.create_index([("org_id", 1), ("agent_type", 1)], name="org_agent_index")
                batch_contacts.create_index([("batch_id", 1), ("status", 1)], name="batch_status_index")
            except Exception as e:
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
        
        logger.info("Database initialization completed successfully")
        logger.info("Collections verified: UserTable, AgentConfig, Audience, Campaigns, CallLogs, PhoneNumber, Members, Integrations, KnowledgeDocuments, Batches, BatchContacts")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
