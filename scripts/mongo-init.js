// =============================================================================
// AI RESEARCH ASSISTANT - MONGODB INITIALIZATION SCRIPT
// =============================================================================

// Switch to the application database
db = db.getSiblingDB('ai_research_assistant');

// Create collections with validation
db.createCollection("research_tasks", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["id", "title", "status", "created_at"],
         properties: {
            id: {
               bsonType: "string",
               description: "Unique identifier for the research task"
            },
            title: {
               bsonType: "string",
               description: "Title of the research task"
            },
            description: {
               bsonType: "string",
               description: "Detailed description of the research task"
            },
            status: {
               bsonType: "string",
               enum: ["pending", "researching", "analyzing", "writing", "completed", "failed"],
               description: "Current status of the task"
            },
            priority: {
               bsonType: "string",
               enum: ["low", "medium", "high", "urgent"],
               description: "Priority level of the task"
            },
            created_at: {
               bsonType: "date",
               description: "Task creation timestamp"
            },
            updated_at: {
               bsonType: "date",
               description: "Task last update timestamp"
            },
            completed_at: {
               bsonType: ["date", "null"],
               description: "Task completion timestamp"
            }
         }
      }
   }
});

db.createCollection("agent_results", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["id", "task_id", "agent_type", "result", "created_at"],
         properties: {
            id: {
               bsonType: "string",
               description: "Unique identifier for the agent result"
            },
            task_id: {
               bsonType: "string",
               description: "Reference to the research task"
            },
            agent_type: {
               bsonType: "string",
               enum: ["researcher", "analyst", "writer"],
               description: "Type of agent that produced the result"
            },
            result: {
               bsonType: "object",
               description: "Agent execution result data"
            },
            metadata: {
               bsonType: "object",
               description: "Additional metadata about the execution"
            },
            execution_time: {
               bsonType: "number",
               description: "Time taken to execute in seconds"
            },
            created_at: {
               bsonType: "date",
               description: "Result creation timestamp"
            }
         }
      }
   }
});

db.createCollection("user_sessions", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["id", "user_id", "created_at"],
         properties: {
            id: {
               bsonType: "string",
               description: "Unique session identifier"
            },
            user_id: {
               bsonType: "string",
               description: "User identifier"
            },
            data: {
               bsonType: "object",
               description: "Session data"
            },
            expires_at: {
               bsonType: "date",
               description: "Session expiration timestamp"
            },
            created_at: {
               bsonType: "date",
               description: "Session creation timestamp"
            }
         }
      }
   }
});

// Create indexes for better performance
db.research_tasks.createIndex({ "id": 1 }, { unique: true });
db.research_tasks.createIndex({ "status": 1 });
db.research_tasks.createIndex({ "created_at": -1 });
db.research_tasks.createIndex({ "priority": 1, "status": 1 });

db.agent_results.createIndex({ "id": 1 }, { unique: true });
db.agent_results.createIndex({ "task_id": 1 });
db.agent_results.createIndex({ "agent_type": 1 });
db.agent_results.createIndex({ "created_at": -1 });

db.user_sessions.createIndex({ "id": 1 }, { unique: true });
db.user_sessions.createIndex({ "user_id": 1 });
db.user_sessions.createIndex({ "expires_at": 1 }, { expireAfterSeconds: 0 });

// Create default admin user (for development only)
db.users.insertOne({
   id: "admin-user",
   username: "admin",
   email: "admin@ai-research.local",
   role: "admin",
   created_at: new Date(),
   is_active: true
});

print("MongoDB initialization completed successfully!");
print("Collections created: research_tasks, agent_results, user_sessions");
print("Indexes created for optimal performance");
print("Default admin user created (development only)");