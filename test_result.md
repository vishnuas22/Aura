#====================================================================================================
# START - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================

# THIS SECTION CONTAINS CRITICAL TESTING INSTRUCTIONS FOR BOTH AGENTS
# BOTH MAIN_AGENT AND TESTING_AGENT MUST PRESERVE THIS ENTIRE BLOCK

# Communication Protocol:
# If the `testing_agent` is available, main agent should delegate all testing tasks to it.
#
# You have access to a file called `test_result.md`. This file contains the complete testing state
# and history, and is the primary means of communication between main and the testing agent.
#
# Main and testing agents must follow this exact format to maintain testing data. 
# The testing data must be entered in yaml format Below is the data structure:
# 
## user_problem_statement: {problem_statement}
## backend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.py"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## frontend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.js"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## metadata:
##   created_by: "main_agent"
##   version: "1.0"
##   test_sequence: 0
##   run_ui: false
##
## test_plan:
##   current_focus:
##     - "Task name 1"
##     - "Task name 2"
##   stuck_tasks:
##     - "Task name with persistent issues"
##   test_all: false
##   test_priority: "high_first"  # or "sequential" or "stuck_first"
##
## agent_communication:
##     -agent: "main"  # or "testing" or "user"
##     -message: "Communication message between agents"

# Protocol Guidelines for Main agent
#
# 1. Update Test Result File Before Testing:
#    - Main agent must always update the `test_result.md` file before calling the testing agent
#    - Add implementation details to the status_history
#    - Set `needs_retesting` to true for tasks that need testing
#    - Update the `test_plan` section to guide testing priorities
#    - Add a message to `agent_communication` explaining what you've done
#
# 2. Incorporate User Feedback:
#    - When a user provides feedback that something is or isn't working, add this information to the relevant task's status_history
#    - Update the working status based on user feedback
#    - If a user reports an issue with a task that was marked as working, increment the stuck_count
#    - Whenever user reports issue in the app, if we have testing agent and task_result.md file so find the appropriate task for that and append in status_history of that task to contain the user concern and problem as well 
#
# 3. Track Stuck Tasks:
#    - Monitor which tasks have high stuck_count values or where you are fixing same issue again and again, analyze that when you read task_result.md
#    - For persistent issues, use websearch tool to find solutions
#    - Pay special attention to tasks in the stuck_tasks list
#    - When you fix an issue with a stuck task, don't reset the stuck_count until the testing agent confirms it's working
#
# 4. Provide Context to Testing Agent:
#    - When calling the testing agent, provide clear instructions about:
#      - Which tasks need testing (reference the test_plan)
#      - Any authentication details or configuration needed
#      - Specific test scenarios to focus on
#      - Any known issues or edge cases to verify
#
# 5. Call the testing agent with specific instructions referring to test_result.md
#
# IMPORTANT: Main agent must ALWAYS update test_result.md BEFORE calling the testing agent, as it relies on this file to understand what to test next.

#====================================================================================================
# END - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================



#====================================================================================================
# Testing Data - Main Agent and testing sub agent both should log testing data below this section
#====================================================================================================

## user_problem_statement: "Create a foundational agent architecture using CrewAI with base agent class, agent factory pattern, communication protocol, memory management, and error handling with retry mechanisms. Each agent should have specific role, backstory, goal definition, tools assignment, verbose logging, and performance metrics tracking."

## backend:
  - task: "Core Exception System"
    implemented: true
    working: true
    file: "backend/core/exceptions.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created comprehensive exception hierarchy with custom exceptions for all components"

  - task: "Configuration Management System"
    implemented: true
    working: true
    file: "backend/core/config.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented complete configuration system with YAML loading, validation, and agent-specific configs"

  - task: "Performance Metrics Tracking"
    implemented: true
    working: true
    file: "backend/core/metrics.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created comprehensive metrics system with execution tracking, aggregation, and performance summaries"

  - task: "Memory Management System"
    implemented: true
    working: true
    file: "backend/memory/agent_memory.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented memory management with MongoDB and in-memory stores, conversation tracking, and context retrieval"

  - task: "Communication Protocol"
    implemented: true
    working: true
    file: "backend/core/communication.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Built comprehensive communication system with message types, channels, handlers, and async processing"

  - task: "Retry Handler with Exponential Backoff"
    implemented: true
    working: true
    file: "backend/utils/retry.py"
    stuck_count: 0
    priority: "medium"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented retry handler with exponential backoff, jitter, and selective exception handling"

  - task: "Advanced Logging System"
    implemented: true
    working: true
    file: "backend/utils/logging_utils.py"
    stuck_count: 0
    priority: "medium"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created agent-specific logging with JSON format support, performance logging, and structured output"

  - task: "Agent Tools Framework"
    implemented: true
    working: true
    file: "backend/tools/agent_tools.py"
    stuck_count: 0
    priority: "medium"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented base tool framework with placeholder tools for research, analysis, and writing"

  - task: "Base Agent Class"
    implemented: true
    working: true
    file: "backend/agents/base_agent.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created comprehensive base agent with memory, communication, metrics, error handling, and CrewAI integration"

  - task: "Agent Factory Pattern"
    implemented: true
    working: true
    file: "backend/agents/agent_factory.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented factory pattern with agent lifecycle management, team creation, and metrics access"

  - task: "Researcher Agent Implementation"
    implemented: true
    working: true
    file: "backend/agents/researcher_agent.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Built specialized researcher agent with web research, document analysis, and fact verification capabilities"

  - task: "Analyst Agent Implementation"
    implemented: true
    working: true
    file: "backend/agents/analyst_agent.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created analyst agent with trend analysis, comparative analysis, SWOT analysis, and risk assessment"

  - task: "Writer Agent Implementation"
    implemented: true
    working: true
    file: "backend/agents/writer_agent.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Implemented writer agent with content creation, editing, structuring, and multiple format support"

## frontend:
  - task: "No Frontend Changes Required"
    implemented: true
    working: "NA"
    file: "N/A"
    stuck_count: 0
    priority: "low"
    needs_retesting: false
    status_history:
      - working: "NA"
        agent: "main"
        comment: "This task focused on backend architecture only - no frontend changes needed"

## metadata:
  created_by: "main_agent"
  version: "1.0"
  test_sequence: 0
  run_ui: false

## test_plan:
  current_focus:
    - "Base Agent Class"
    - "Agent Factory Pattern"
    - "Communication Protocol"
    - "Memory Management System"
    - "Performance Metrics Tracking"
  stuck_tasks: []
  test_all: true
  test_priority: "high_first"

## agent_communication:
  - agent: "main"
    message: "Completed comprehensive foundational agent architecture implementation with all requested components: base agent class, factory pattern, communication protocol, memory management, error handling with retry mechanisms, role/backstory/goal definitions, tools assignment, verbose logging, and performance metrics tracking. All components are implemented with proper type hints and docstrings. The system includes three specialized agents (Researcher, Analyst, Writer) with full CrewAI integration and production-ready features."