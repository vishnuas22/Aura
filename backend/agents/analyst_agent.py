"""
Analyst Agent - Specialized in data analysis and strategic insights.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import statistics

from .base_agent import BaseAIAgent
from ..core.config import AgentConfig
from ..core.exceptions import AgentException
from ..tools.agent_tools import get_tools_for_agent
from ..utils.logging_utils import log_agent_performance, log_agent_task


class AnalystAgent(BaseAIAgent):
    """
    Analyst Agent specializing in data analysis and strategic insights.
    
    Capabilities:
    - Data analysis and pattern recognition
    - Trend analysis and forecasting
    - Strategic recommendations
    - Comparative analysis
    - SWOT analysis
    - Risk assessment
    """
    
    def __init__(
        self, 
        agent_type: str = "analyst",
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        memory_store_type: str = "memory",
        tools: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize Analyst Agent.
        
        Args:
            agent_type: Agent type (defaults to "analyst")
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
            tools = asyncio.run(get_tools_for_agent("analyst"))
        
        self._init_crew_agent(tools)
        
        # Analysis-specific attributes
        self.analysis_frameworks = [
            "SWOT", "PEST", "5_Forces", "Value_Chain", "BCG_Matrix"
        ]
        self.confidence_threshold = 0.6
        self.min_data_points = 3
        
        self.logger.info(f"Analyst agent initialized: {self.agent_id}")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an analysis task.
        
        Args:
            task_data: Task data containing analysis requirements
            
        Returns:
            Analysis results and insights
        """
        start_time = datetime.utcnow()
        task_type = task_data.get("type", "general_analysis")
        
        try:
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "started")
            
            # Extract task parameters
            data_source = task_data.get("data", {})
            analysis_framework = task_data.get("framework", "comprehensive")
            focus_areas = task_data.get("focus_areas", [])
            confidence_required = task_data.get("confidence_threshold", self.confidence_threshold)
            
            if not data_source and task_type != "synthesis":
                raise AgentException("Data source is required for analysis", error_code="MISSING_DATA")
            
            self.logger.info(f"Starting analysis task: {task_type} using {analysis_framework} framework")
            
            # Store analysis start in memory
            await self.memory.store_conversation(
                message=f"Starting {task_type} analysis using {analysis_framework} framework",
                role="system",
                metadata={
                    "task_type": task_type,
                    "framework": analysis_framework,
                    "focus_areas": focus_areas
                }
            )
            
            # Execute analysis based on task type
            if task_type == "trend_analysis":
                result = await self._perform_trend_analysis(
                    data=data_source,
                    focus_areas=focus_areas
                )
            elif task_type == "comparative_analysis":
                result = await self._perform_comparative_analysis(
                    data=data_source,
                    comparison_criteria=task_data.get("criteria", [])
                )
            elif task_type == "swot_analysis":
                result = await self._perform_swot_analysis(
                    context=data_source,
                    subject=task_data.get("subject", "Unknown")
                )
            elif task_type == "risk_assessment":
                result = await self._perform_risk_assessment(
                    data=data_source,
                    risk_categories=task_data.get("risk_categories", [])
                )
            elif task_type == "synthesis":
                # Synthesize insights from memory and provided data
                result = await self._synthesize_insights(
                    query=task_data.get("query", ""),
                    additional_data=data_source
                )
            else:
                # General comprehensive analysis
                result = await self._perform_comprehensive_analysis(
                    data=data_source,
                    framework=analysis_framework,
                    focus_areas=focus_areas
                )
            
            # Validate confidence level
            if result.get("confidence", 0) < confidence_required:
                self.logger.warning(f"Analysis confidence {result.get('confidence', 0):.2f} below required {confidence_required:.2f}")
                result["warning"] = f"Analysis confidence below required threshold"
            
            # Store analysis results in memory
            await self.memory.store_insight(
                insight=f"Analysis completed: {result.get('summary', 'No summary available')}",
                category="analysis_completion",
                confidence=result.get('confidence', 0.5),
                metadata={
                    "analysis_type": task_type,
                    "framework": analysis_framework,
                    "insights_count": len(result.get('insights', [])),
                    "recommendations_count": len(result.get('recommendations', [])),
                    "analysis_duration": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            # Add metadata to result
            result.update({
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "task_id": task_data.get("id"),
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": datetime.utcnow().isoformat(),
                "framework_used": analysis_framework,
                "tools_used": [tool.name for tool in self._tools if hasattr(tool, 'name')]
            })
            
            # Log performance metrics
            log_agent_performance(
                self.logger,
                self.agent_id,
                f"analysis_{task_type}",
                result["execution_time"],
                success=True,
                metadata={
                    "insights_generated": len(result.get('insights', [])),
                    "confidence": result.get('confidence', 0)
                }
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "completed")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store failure in memory
            await self.memory.store_conversation(
                message=f"Analysis task failed: {str(e)}",
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
                f"analysis_{task_type}",
                execution_time,
                success=False,
                metadata={"error": str(e)}
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "failed")
            
            raise
    
    async def _perform_trend_analysis(
        self, 
        data: Dict[str, Any], 
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Perform trend analysis on provided data."""
        self.logger.info("Performing trend analysis")
        
        # Get data analysis tool
        analysis_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'data_analysis'), None)
        
        if not analysis_tool:
            raise AgentException("Data analysis tool not available", error_code="TOOL_NOT_AVAILABLE")
        
        # Execute trend analysis
        analysis_result = await analysis_tool.execute(data=data, analysis_type="trend_analysis")
        
        # Process and enhance results
        trends = self._extract_trends(data, focus_areas)
        forecasts = self._generate_forecasts(trends)
        
        insights = [
            f"Identified {len(trends)} significant trends",
            f"Generated forecasts for {len(forecasts)} trend categories",
            f"Analysis focused on {len(focus_areas)} key areas" if focus_areas else "Comprehensive trend analysis performed"
        ]
        
        recommendations = [
            "Monitor identified trends closely for strategic decision-making",
            "Consider early action on high-confidence forecasts",
            "Review trend analysis quarterly for accuracy validation"
        ]
        
        return {
            "analysis_type": "trend_analysis",
            "trends_identified": trends,
            "forecasts": forecasts,
            "focus_areas": focus_areas,
            "insights": insights,
            "recommendations": recommendations,
            "summary": f"Trend analysis identified {len(trends)} key trends with varying confidence levels",
            "confidence": self._calculate_analysis_confidence(trends),
            "methodology": "statistical_trend_analysis_with_forecasting"
        }
    
    async def _perform_comparative_analysis(
        self, 
        data: Dict[str, Any], 
        comparison_criteria: List[str]
    ) -> Dict[str, Any]:
        """Perform comparative analysis."""
        self.logger.info("Performing comparative analysis")
        
        if not comparison_criteria:
            comparison_criteria = ["performance", "efficiency", "cost", "quality"]
        
        comparisons = []
        for criterion in comparison_criteria:
            comparison = self._analyze_criterion(data, criterion)
            comparisons.append(comparison)
        
        # Generate comparative insights
        insights = []
        for comp in comparisons:
            if comp.get("difference", 0) > 0.2:  # 20% threshold
                insights.append(f"Significant difference in {comp['criterion']}: {comp['summary']}")
        
        # Generate recommendations based on comparisons
        recommendations = []
        top_performer = max(comparisons, key=lambda x: x.get("score", 0))
        recommendations.append(f"Consider adopting best practices from {top_performer['criterion']} analysis")
        
        return {
            "analysis_type": "comparative_analysis",
            "comparison_criteria": comparison_criteria,
            "detailed_comparisons": comparisons,
            "insights": insights,
            "recommendations": recommendations,
            "summary": f"Comparative analysis across {len(comparison_criteria)} criteria reveals {len(insights)} key differences",
            "confidence": statistics.mean([comp.get("confidence", 0.5) for comp in comparisons]),
            "methodology": "multi_criteria_comparative_analysis"
        }
    
    async def _perform_swot_analysis(
        self, 
        context: Dict[str, Any], 
        subject: str
    ) -> Dict[str, Any]:
        """Perform SWOT analysis."""
        self.logger.info(f"Performing SWOT analysis for: {subject}")
        
        # Extract SWOT elements from context (simplified implementation)
        swot = {
            "strengths": self._extract_swot_elements(context, "strengths"),
            "weaknesses": self._extract_swot_elements(context, "weaknesses"),
            "opportunities": self._extract_swot_elements(context, "opportunities"),
            "threats": self._extract_swot_elements(context, "threats")
        }
        
        # Generate strategic insights
        insights = []
        for category, items in swot.items():
            if items:
                insights.append(f"Identified {len(items)} key {category}")
        
        # Strategic recommendations
        recommendations = [
            "Leverage strengths to capitalize on opportunities",
            "Address weaknesses that could become vulnerabilities",
            "Develop contingency plans for identified threats",
            "Create action plans for high-priority opportunities"
        ]
        
        return {
            "analysis_type": "swot_analysis",
            "subject": subject,
            "swot_matrix": swot,
            "insights": insights,
            "recommendations": recommendations,
            "summary": f"SWOT analysis for {subject} identified {sum(len(v) for v in swot.values())} strategic factors",
            "confidence": 0.8,  # SWOT is generally high-confidence when well-structured
            "methodology": "structured_swot_framework_analysis"
        }
    
    async def _perform_risk_assessment(
        self, 
        data: Dict[str, Any], 
        risk_categories: List[str]
    ) -> Dict[str, Any]:
        """Perform risk assessment."""
        self.logger.info("Performing risk assessment")
        
        if not risk_categories:
            risk_categories = ["operational", "financial", "strategic", "compliance", "technology"]
        
        risk_analysis = {}
        total_risk_score = 0
        
        for category in risk_categories:
            risk_score = self._assess_risk_category(data, category)
            risk_analysis[category] = risk_score
            total_risk_score += risk_score["score"]
        
        # Calculate overall risk level
        avg_risk_score = total_risk_score / len(risk_categories)
        risk_level = "high" if avg_risk_score > 0.7 else "medium" if avg_risk_score > 0.4 else "low"
        
        # Generate risk insights
        insights = [
            f"Overall risk level assessed as {risk_level}",
            f"Highest risk area: {max(risk_analysis.items(), key=lambda x: x[1]['score'])[0]}",
            f"Risk assessment covers {len(risk_categories)} key categories"
        ]
        
        # Risk mitigation recommendations
        high_risk_areas = [k for k, v in risk_analysis.items() if v["score"] > 0.7]
        recommendations = [
            f"Prioritize mitigation for high-risk areas: {', '.join(high_risk_areas)}" if high_risk_areas else "Continue monitoring current risk levels",
            "Implement regular risk monitoring and review processes",
            "Develop contingency plans for identified high-probability risks"
        ]
        
        return {
            "analysis_type": "risk_assessment",
            "risk_categories": risk_categories,
            "risk_analysis": risk_analysis,
            "overall_risk_level": risk_level,
            "average_risk_score": avg_risk_score,
            "insights": insights,
            "recommendations": recommendations,
            "summary": f"Risk assessment identifies {risk_level} overall risk with {len(high_risk_areas)} high-risk areas",
            "confidence": 0.75,
            "methodology": "multi_category_risk_scoring_framework"
        }
    
    async def _synthesize_insights(
        self, 
        query: str, 
        additional_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize insights from memory and additional data."""
        self.logger.info(f"Synthesizing insights for: {query}")
        
        # Get relevant insights from memory
        relevant_context = await self.memory.get_relevant_context(query, limit=10)
        past_insights = await self.memory.get_recent_insights(limit=15)
        
        # Extract insights
        contextual_insights = [item.content.get("insight", "") for item in relevant_context]
        historical_insights = [item.content.get("insight", "") for item in past_insights]
        
        # Combine with additional data
        synthesis_data = {
            "contextual_insights": contextual_insights,
            "historical_insights": historical_insights,
            "additional_data": additional_data,
            "query_focus": query
        }
        
        # Perform synthesis analysis
        synthesized_insights = self._synthesize_data_points(synthesis_data)
        patterns = self._identify_synthesis_patterns(synthesis_data)
        
        recommendations = [
            "Consider patterns identified across multiple data sources",
            "Validate synthesized insights against current objectives",
            "Monitor for confirmation or contradiction of synthesized patterns"
        ]
        
        return {
            "analysis_type": "insight_synthesis",
            "query": query,
            "data_sources_used": len(contextual_insights) + len(historical_insights),
            "synthesized_insights": synthesized_insights,
            "identified_patterns": patterns,
            "insights": [f"Synthesized {len(synthesized_insights)} high-level insights from multiple sources"],
            "recommendations": recommendations,
            "summary": f"Insight synthesis for '{query}' combines {len(contextual_insights) + len(historical_insights)} data points into {len(synthesized_insights)} key insights",
            "confidence": 0.7,  # Synthesis confidence depends on source quality
            "methodology": "multi_source_insight_synthesis"
        }
    
    async def _perform_comprehensive_analysis(
        self, 
        data: Dict[str, Any], 
        framework: str,
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis using specified framework."""
        self.logger.info(f"Performing comprehensive analysis using {framework} framework")
        
        analysis_components = []
        
        # Include multiple analysis types
        if "trend" in framework.lower() or "comprehensive" in framework.lower():
            trend_analysis = await self._perform_trend_analysis(data, focus_areas)
            analysis_components.append(("trend_analysis", trend_analysis))
        
        if "comparative" in framework.lower() or "comprehensive" in framework.lower():
            comparative_analysis = await self._perform_comparative_analysis(data, [])
            analysis_components.append(("comparative_analysis", comparative_analysis))
        
        if "risk" in framework.lower() or "comprehensive" in framework.lower():
            risk_analysis = await self._perform_risk_assessment(data, [])
            analysis_components.append(("risk_assessment", risk_analysis))
        
        # Combine insights from all components
        all_insights = []
        all_recommendations = []
        confidence_scores = []
        
        for component_name, component_result in analysis_components:
            all_insights.extend(component_result.get("insights", []))
            all_recommendations.extend(component_result.get("recommendations", []))
            confidence_scores.append(component_result.get("confidence", 0.5))
        
        # Calculate overall confidence
        overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5
        
        return {
            "analysis_type": "comprehensive_analysis",
            "framework": framework,
            "focus_areas": focus_areas,
            "analysis_components": [comp[0] for comp in analysis_components],
            "detailed_analyses": {comp[0]: comp[1] for comp in analysis_components},
            "insights": all_insights,
            "recommendations": all_recommendations,
            "summary": f"Comprehensive {framework} analysis completed with {len(analysis_components)} component analyses",
            "confidence": overall_confidence,
            "methodology": f"{framework}_multi_component_analysis"
        }
    
    def _extract_trends(self, data: Dict[str, Any], focus_areas: List[str]) -> List[Dict[str, Any]]:
        """Extract trends from data (simplified implementation)."""
        trends = []
        
        # Simplified trend extraction
        for area in focus_areas or ["general"]:
            trend = {
                "area": area,
                "direction": "upward",  # Placeholder
                "strength": 0.7,
                "confidence": 0.8,
                "time_horizon": "3-6 months",
                "description": f"Trending pattern identified in {area}"
            }
            trends.append(trend)
        
        return trends
    
    def _generate_forecasts(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate forecasts based on identified trends."""
        forecasts = []
        
        for trend in trends:
            forecast = {
                "area": trend["area"],
                "prediction": f"Continued {trend['direction']} trajectory",
                "confidence": trend["confidence"] * 0.8,  # Forecasts are less certain
                "time_horizon": "6-12 months",
                "factors": ["current trend strength", "historical patterns"]
            }
            forecasts.append(forecast)
        
        return forecasts
    
    def _calculate_analysis_confidence(self, analysis_data: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for analysis."""
        if not analysis_data:
            return 0.3
        
        confidence_scores = [item.get("confidence", 0.5) for item in analysis_data]
        data_points = len(analysis_data)
        
        # Adjust confidence based on number of data points
        base_confidence = statistics.mean(confidence_scores)
        data_confidence_factor = min(1.0, data_points / self.min_data_points)
        
        return base_confidence * data_confidence_factor
    
    def _analyze_criterion(self, data: Dict[str, Any], criterion: str) -> Dict[str, Any]:
        """Analyze a specific criterion (simplified implementation)."""
        return {
            "criterion": criterion,
            "score": 0.75,  # Placeholder score
            "confidence": 0.8,
            "difference": 0.15,  # Difference from baseline
            "summary": f"{criterion} analysis shows above-average performance"
        }
    
    def _extract_swot_elements(self, context: Dict[str, Any], category: str) -> List[str]:
        """Extract SWOT elements from context (simplified implementation)."""
        # Placeholder implementation - in real scenario, this would analyze the context
        elements_map = {
            "strengths": ["Strong market position", "Experienced team", "Proven track record"],
            "weaknesses": ["Limited resources", "Market dependence", "Technology gaps"],
            "opportunities": ["Market expansion", "New technologies", "Strategic partnerships"],
            "threats": ["Increased competition", "Market volatility", "Regulatory changes"]
        }
        
        return elements_map.get(category, [])[:2]  # Return subset for demo
    
    def _assess_risk_category(self, data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Assess risk for a specific category (simplified implementation)."""
        # Placeholder risk scores
        risk_scores = {
            "operational": 0.4,
            "financial": 0.6,
            "strategic": 0.3,
            "compliance": 0.5,
            "technology": 0.7
        }
        
        score = risk_scores.get(category, 0.5)
        
        return {
            "category": category,
            "score": score,
            "level": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
            "factors": [f"Factor 1 for {category}", f"Factor 2 for {category}"],
            "mitigation_priority": "high" if score > 0.6 else "medium"
        }
    
    def _synthesize_data_points(self, synthesis_data: Dict[str, Any]) -> List[str]:
        """Synthesize insights from multiple data points."""
        # Simplified synthesis - in real implementation, this would use advanced analysis
        insights_count = len(synthesis_data.get("contextual_insights", []))
        historical_count = len(synthesis_data.get("historical_insights", []))
        
        synthesized = [
            f"Pattern identified across {insights_count} contextual data points",
            f"Historical analysis of {historical_count} previous insights reveals consistency",
            f"Combined analysis suggests high confidence in synthesized conclusions"
        ]
        
        return synthesized
    
    def _identify_synthesis_patterns(self, synthesis_data: Dict[str, Any]) -> List[str]:
        """Identify patterns in synthesized data."""
        return [
            "Recurring themes across multiple time periods",
            "Consistent performance indicators",
            "Predictable response patterns to market changes"
        ]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools for analyst agent."""
        return [
            "data_analysis",
            "pattern_recognition",
            "trend_analysis",
            "statistical_analysis",
            "comparative_analysis",
            "swot_analysis",
            "risk_assessment"
        ]
    
    async def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent analysis history.
        
        Args:
            limit: Maximum number of analysis entries to return
            
        Returns:
            List of analysis history entries
        """
        insights = await self.memory.get_recent_insights("analysis_completion", limit=limit)
        
        history = []
        for insight in insights:
            history.append({
                "timestamp": insight.timestamp.isoformat(),
                "analysis_type": insight.metadata.get("analysis_type", "unknown"),
                "framework": insight.metadata.get("framework", "unknown"),
                "insights_count": insight.metadata.get("insights_count", 0),
                "confidence": insight.importance,
                "summary": insight.content.get("insight", "")
            })
        
        return history