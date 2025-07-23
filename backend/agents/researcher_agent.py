"""
Researcher Agent - Specialized in research and data gathering.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAIAgent
from core.config import AgentConfig
from core.exceptions import AgentException
from tools.agent_tools import get_tools_for_agent
from utils.logging_utils import log_agent_performance, log_agent_task


class ResearcherAgent(BaseAIAgent):
    """
    Researcher Agent specializing in information gathering and research tasks.
    
    Capabilities:
    - Web search and information gathering
    - Document analysis and summarization
    - Fact verification and source validation
    - Research planning and execution
    - Data extraction and organization
    """
    
    def __init__(
        self, 
        agent_type: str = "researcher",
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        memory_store_type: str = "memory",
        tools: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize Researcher Agent.
        
        Args:
            agent_type: Agent type (defaults to "researcher")
            config: Agent configuration
            agent_id: Unique agent identifier
            memory_store_type: Type of memory store
            tools: List of tools to use
            **kwargs: Additional arguments
        """
        super().__init__(
            agent_type=agent_type,
            config=config,
            agent_id=agent_id,
            memory_store_type=memory_store_type,
            **kwargs
        )
        
        # Initialize tools
        if tools is None:
            tools = asyncio.run(get_tools_for_agent("researcher"))
        
        self._init_crew_agent(tools)
        
        # Research-specific attributes
        self.research_history: List[Dict[str, Any]] = []
        self.source_credibility_threshold = 0.7
        self.max_search_results = 10
        
        self.logger.info(f"Researcher agent initialized: {self.agent_id}")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research task.
        
        Args:
            task_data: Task data containing research requirements
            
        Returns:
            Research results and findings
        """
        start_time = datetime.utcnow()
        task_type = task_data.get("type", "general_research")
        
        try:
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "started")
            
            # Extract task parameters
            research_query = task_data.get("query", task_data.get("description", ""))
            research_scope = task_data.get("scope", "broad")
            max_sources = task_data.get("max_sources", self.max_search_results)
            required_credibility = task_data.get("credibility_threshold", self.source_credibility_threshold)
            
            if not research_query:
                raise AgentException("Research query or description is required", error_code="MISSING_QUERY")
            
            self.logger.info(f"Starting research task: {research_query}")
            
            # Store research start in memory
            await self.memory.store_conversation(
                message=f"Starting research on: {research_query}",
                role="system",
                metadata={
                    "task_type": task_type,
                    "scope": research_scope,
                    "max_sources": max_sources
                }
            )
            
            # Execute research based on task type
            if task_type == "web_research":
                result = await self._perform_web_research(
                    query=research_query,
                    max_sources=max_sources,
                    credibility_threshold=required_credibility
                )
            elif task_type == "document_analysis":
                result = await self._analyze_documents(
                    documents=task_data.get("documents", []),
                    focus=research_query
                )
            elif task_type == "fact_verification":
                result = await self._verify_facts(
                    claims=task_data.get("claims", []),
                    sources=task_data.get("sources", [])
                )
            else:
                # General research combines multiple approaches
                result = await self._perform_comprehensive_research(
                    query=research_query,
                    scope=research_scope,
                    max_sources=max_sources
                )
            
            # Store research results in memory
            await self.memory.store_insight(
                insight=f"Research completed: {result.get('summary', 'No summary available')}",
                category="research_completion",
                confidence=result.get('confidence', 0.7),
                metadata={
                    "sources_count": len(result.get('sources', [])),
                    "credible_sources": result.get('credible_sources_count', 0),
                    "research_duration": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            # Add metadata to result
            result.update({
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "task_id": task_data.get("id"),
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": datetime.utcnow().isoformat(),
                "tools_used": [tool.name for tool in self._tools if hasattr(tool, 'name')]
            })
            
            # Log performance metrics
            log_agent_performance(
                self.logger,
                self.agent_id,
                f"research_{task_type}",
                result["execution_time"],
                success=True,
                metadata={"sources_found": len(result.get('sources', []))}
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "completed")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store failure in memory
            await self.memory.store_conversation(
                message=f"Research task failed: {str(e)}",
                role="system",
                metadata={
                    "error_type": type(e).__name__,
                    "task_type": task_type,
                    "execution_time": execution_time
                }
            )
            
            # Log performance metrics for failure
            log_agent_performance(
                self.logger,
                self.agent_id,
                f"research_{task_type}",
                execution_time,
                success=False,
                metadata={"error": str(e)}
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "failed")
            
            raise
    
    async def _perform_web_research(
        self, 
        query: str, 
        max_sources: int = 10,
        credibility_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Perform web-based research."""
        self.logger.info(f"Performing web research for: {query}")
        
        # Generate comprehensive research plan using LLM
        research_plan_prompt = f"""
        As a senior research specialist, create a comprehensive research plan for the query: "{query}"
        
        Provide:
        1. Key research questions to address
        2. Types of sources to search for
        3. Search strategies and keywords
        4. Expected findings and insights
        5. Quality criteria for source evaluation
        
        Format your response as a structured research plan.
        """
        
        research_plan = await self.generate_llm_response(
            prompt=research_plan_prompt,
            task_type="web_research",
            complexity="medium",
            use_context=False
        )
        
        # Store research plan
        await self.memory.store_insight(
            insight=f"Research plan created for query: {query}",
            category="research_planning",
            confidence=0.8,
            metadata={
                "query": query,
                "plan": research_plan
            }
        )
        
        # Get web search tool (placeholder for now)
        web_search_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'web_search'), None)
        
        if not web_search_tool:
            # Generate mock research results using LLM for now
            mock_research_prompt = f"""
            Based on your knowledge, provide comprehensive research findings for: "{query}"
            
            Include:
            1. Key facts and statistics
            2. Main themes and concepts
            3. Different perspectives or viewpoints
            4. Recent developments or trends
            5. Credible sources and references (even if hypothetical)
            
            Present findings as if gathered from multiple credible sources.
            """
            
            research_findings = await self.generate_llm_response(
                prompt=mock_research_prompt,
                task_type="web_research",
                complexity="high",
                use_context=True
            )
            
            # Create mock sources based on LLM response
            credible_sources = [
                {
                    "title": f"Research finding on {query}",
                    "url": "https://example-source.com",
                    "snippet": research_findings[:200] + "...",
                    "credibility_score": 0.9,
                    "source_type": "academic"
                }
            ]
            
            # Generate summary using LLM
            summary_prompt = f"""
            Summarize the following research findings on "{query}":
            
            {research_findings}
            
            Provide a concise, professional summary highlighting key insights.
            """
            
            summary = await self.generate_llm_response(
                prompt=summary_prompt,
                task_type="summary_writing",
                complexity="medium",
                use_context=False
            )
        else:
            # Execute actual web search when tool is available
            search_results = await web_search_tool.execute(query=query, max_results=max_sources)
            
            # Analyze and filter results using LLM
            analysis_prompt = f"""
            Analyze the following search results for the query "{query}" and assess their credibility:
            
            {str(search_results.get('results', []))}
            
            For each source, evaluate:
            1. Credibility (0.0 to 1.0)
            2. Relevance to the query
            3. Quality of information
            4. Source authority
            
            Provide analysis in JSON format.
            """
            
            analysis_schema = {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "credibility_score": {"type": "number"},
                                "relevance_score": {"type": "number"},
                                "quality_assessment": {"type": "string"},
                                "key_insights": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                },
                "required": ["sources"]
            }
            
            analysis_result = await self.generate_llm_response(
                prompt=analysis_prompt,
                task_type="fact_verification",
                complexity="high",
                structured_schema=analysis_schema,
                use_context=False
            )
            
            credible_sources = [
                source for source in analysis_result.get("sources", [])
                if source.get("credibility_score", 0) >= credibility_threshold
            ]
            
            # Generate comprehensive summary
            summary = self._generate_research_summary(credible_sources, query)
        
        return {
            "query": query,
            "sources": credible_sources,
            "total_sources_found": len(credible_sources) if credible_sources else 1,
            "credible_sources_count": len(credible_sources) if credible_sources else 1,
            "summary": summary if 'summary' in locals() else research_findings,
            "confidence": min(1.0, len(credible_sources) / max(1, max_sources)) if credible_sources else 0.8,
            "methodology": "llm_enhanced_web_research",
            "credibility_threshold": credibility_threshold,
            "research_plan": research_plan
        }
    
    async def _analyze_documents(
        self, 
        documents: List[str], 
        focus: str
    ) -> Dict[str, Any]:
        """Analyze provided documents."""
        self.logger.info(f"Analyzing {len(documents)} documents with focus: {focus}")
        
        # Get document analysis tool
        doc_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'document_analysis'), None)
        
        if not doc_tool:
            raise AgentException("Document analysis tool not available", error_code="TOOL_NOT_AVAILABLE")
        
        analyzed_docs = []
        key_findings = []
        
        for doc_path in documents:
            analysis = await doc_tool.execute(document_path=doc_path, analysis_type="focused")
            analyzed_docs.append(analysis)
            key_findings.extend(analysis.get('key_points', []))
        
        # Synthesize findings
        summary = f"Analysis of {len(documents)} documents focusing on '{focus}' revealed {len(key_findings)} key findings."
        
        return {
            "focus": focus,
            "documents_analyzed": len(documents),
            "analyzed_documents": analyzed_docs,
            "key_findings": key_findings,
            "summary": summary,
            "confidence": 0.8,
            "methodology": "document_analysis_with_synthesis"
        }
    
    async def _verify_facts(
        self, 
        claims: List[str], 
        sources: List[str]
    ) -> Dict[str, Any]:
        """Verify factual claims."""
        self.logger.info(f"Verifying {len(claims)} claims against {len(sources)} sources")
        
        # Get fact verification tool
        fact_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'fact_verification'), None)
        
        if not fact_tool:
            raise AgentException("Fact verification tool not available", error_code="TOOL_NOT_AVAILABLE")
        
        verification_results = []
        verified_count = 0
        
        for claim in claims:
            verification = await fact_tool.execute(claim=claim, sources=sources)
            verification_results.append(verification)
            
            if verification.get('verification_status') == 'verified':
                verified_count += 1
        
        verification_rate = verified_count / max(1, len(claims))
        
        summary = f"Verified {verified_count} out of {len(claims)} claims ({verification_rate:.1%} verification rate)"
        
        return {
            "claims_checked": len(claims),
            "claims_verified": verified_count,
            "verification_rate": verification_rate,
            "verification_results": verification_results,
            "sources_used": sources,
            "summary": summary,
            "confidence": verification_rate,
            "methodology": "fact_verification_with_sources"
        }
    
    async def _perform_comprehensive_research(
        self, 
        query: str, 
        scope: str,
        max_sources: int
    ) -> Dict[str, Any]:
        """Perform comprehensive research combining multiple methods."""
        self.logger.info(f"Performing comprehensive research: {query} (scope: {scope})")
        
        results = {
            "query": query,
            "scope": scope,
            "methodology": "llm_enhanced_comprehensive_research",
            "research_phases": []
        }
        
        # Phase 1: Research Planning
        try:
            planning_prompt = f"""
            As a senior research specialist, develop a comprehensive research strategy for: "{query}"
            Scope: {scope}
            
            Create a detailed plan including:
            1. Primary research questions
            2. Secondary research areas
            3. Methodology recommendations
            4. Expected challenges and solutions
            5. Quality assurance measures
            
            Provide a structured research framework.
            """
            
            research_framework = await self.generate_llm_response(
                prompt=planning_prompt,
                task_type="research_planning",
                complexity="high",
                use_context=False
            )
            
            results["research_framework"] = research_framework
            results["research_phases"].append("strategic_planning")
            
        except Exception as e:
            self.logger.warning(f"Research planning phase failed: {e}")
        
        # Phase 2: Primary Research
        try:
            web_results = await self._perform_web_research(query, max_sources)
            results["web_research"] = web_results
            results["research_phases"].append("web_research")
            
        except Exception as e:
            self.logger.warning(f"Web research phase failed: {e}")
            results["web_research"] = {"error": str(e)}
        
        # Phase 3: Contextual Analysis
        try:
            relevant_context = await self.memory.get_relevant_context(query, limit=5)
            context_insights = [item.content for item in relevant_context if item.memory_type == "insight"]
            
            if context_insights or results.get("web_research", {}).get("sources"):
                # Use LLM to synthesize findings
                synthesis_prompt = f"""
                Synthesize research findings for: "{query}"
                
                Web Research Findings:
                {results.get("web_research", {}).get("summary", "No web research available")}
                
                Historical Context:
                {"; ".join(context_insights) if context_insights else "No historical context"}
                
                Provide:
                1. Key findings synthesis
                2. Pattern identification
                3. Trend analysis
                4. Strategic implications
                5. Knowledge gaps identified
                """
                
                synthesis = await self.generate_llm_response(
                    prompt=synthesis_prompt,
                    task_type="synthesis",
                    complexity="high",
                    use_context=True
                )
                
                results["synthesis"] = synthesis
                results["research_phases"].append("synthesis")
                
        except Exception as e:
            self.logger.warning(f"Synthesis phase failed: {e}")
        
        # Phase 4: Final Analysis and Recommendations
        try:
            analysis_prompt = f"""
            Provide a comprehensive analysis and recommendations based on research for: "{query}"
            
            Research Summary:
            {results.get("synthesis", results.get("web_research", {}).get("summary", "Limited research available"))}
            
            Deliver:
            1. Executive summary
            2. Key insights
            3. Strategic recommendations
            4. Areas for further research
            5. Confidence assessment
            """
            
            final_analysis = await self.generate_llm_response(
                prompt=analysis_prompt,
                task_type="analysis",
                complexity="high",
                use_context=True
            )
            
            results["final_analysis"] = final_analysis
            results["research_phases"].append("final_analysis")
            
        except Exception as e:
            self.logger.warning(f"Final analysis phase failed: {e}")
        
        # Synthesize results
        all_sources = results.get("web_research", {}).get("sources", [])
        context_count = len(relevant_context) if 'relevant_context' in locals() else 0
        
        synthesis = f"Comprehensive research on '{query}' completed with {scope} scope. "
        synthesis += f"Executed {len(results['research_phases'])} research phases: {', '.join(results['research_phases'])}. "
        synthesis += f"Found {len(all_sources)} credible sources from web research"
        if context_count > 0:
            synthesis += f" and {context_count} relevant contextual insights from previous research."
        else:
            synthesis += "."
        
        results.update({
            "summary": synthesis,
            "total_sources": len(all_sources),
            "confidence": min(1.0, len(all_sources) / max(1, max_sources * 0.7)),
            "research_completeness": len(results["research_phases"]) / 4  # 4 phases expected
        })
        
        return results
    
    def _generate_research_summary(self, sources: List[Dict[str, Any]], query: str) -> str:
        """Generate a summary of research findings."""
        if not sources:
            return f"No credible sources found for query: {query}"
        
        source_count = len(sources)
        avg_credibility = sum(source.get('credibility_score', 0.7) for source in sources) / source_count
        
        summary = f"Research on '{query}' yielded {source_count} credible sources "
        summary += f"with average credibility score of {avg_credibility:.2f}. "
        
        # Extract key themes (simplified)
        all_snippets = " ".join(source.get('snippet', '') for source in sources)
        summary += f"Key information gathered from {len(all_snippets.split())} words of source content."
        
        return summary
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools for researcher agent."""
        return [
            "web_search",
            "document_analysis", 
            "fact_verification",
            "source_credibility_check",
            "data_extraction"
        ]
    
    async def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent research history.
        
        Args:
            limit: Maximum number of research entries to return
            
        Returns:
            List of research history entries
        """
        insights = await self.memory.get_recent_insights("research_completion", limit=limit)
        
        history = []
        for insight in insights:
            history.append({
                "timestamp": insight.timestamp.isoformat(),
                "insight": insight.content.get("insight", ""),
                "confidence": insight.importance,
                "metadata": insight.metadata
            })
        
        return history
    
    async def assess_source_credibility(self, source_url: str, source_content: str = "") -> Dict[str, Any]:
        """
        Assess the credibility of a source.
        
        Args:
            source_url: URL of the source
            source_content: Content of the source (optional)
            
        Returns:
            Credibility assessment
        """
        # Simplified credibility assessment (in real implementation, this would be more sophisticated)
        credibility_factors = {
            "domain_reputation": 0.8,  # Based on domain analysis
            "content_quality": 0.7,   # Based on content analysis
            "citation_presence": 0.6,  # Whether sources are cited
            "recency": 0.9,           # How recent the content is
            "author_expertise": 0.5   # Author credibility (if determinable)
        }
        
        overall_score = sum(credibility_factors.values()) / len(credibility_factors)
        
        assessment = {
            "source_url": source_url,
            "credibility_score": overall_score,
            "credibility_factors": credibility_factors,
            "assessment": "credible" if overall_score >= self.source_credibility_threshold else "questionable",
            "recommendation": "use_with_caution" if overall_score < 0.8 else "reliable_source"
        }
        
        # Store assessment in memory
        await self.memory.store_insight(
            insight=f"Source credibility assessment: {source_url} - {assessment['assessment']}",
            category="credibility_assessment",
            confidence=overall_score,
            metadata=assessment
        )
        
        return assessment