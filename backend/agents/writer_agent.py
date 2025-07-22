"""
Writer Agent - Specialized in content creation and writing tasks.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re

from .base_agent import BaseAIAgent
from ..core.config import AgentConfig
from ..core.exceptions import AgentException
from ..tools.agent_tools import get_tools_for_agent
from ..utils.logging_utils import log_agent_performance, log_agent_task


class WriterAgent(BaseAIAgent):
    """
    Writer Agent specializing in content creation and writing tasks.
    
    Capabilities:
    - Content creation and structuring
    - Document writing and formatting
    - Style optimization and editing
    - Grammar and readability checking
    - Multi-format output generation
    - Citation and reference management
    """
    
    def __init__(
        self, 
        agent_type: str = "writer",
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        memory_store_type: str = "memory",
        tools: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize Writer Agent.
        
        Args:
            agent_type: Agent type (defaults to "writer")
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
            tools = asyncio.run(get_tools_for_agent("writer"))
        
        self._init_crew_agent(tools)
        
        # Writing-specific attributes
        self.supported_formats = ["markdown", "html", "pdf", "docx", "plain_text"]
        self.writing_styles = {
            "academic": {"tone": "formal", "complexity": "high", "citations": True},
            "business": {"tone": "professional", "complexity": "medium", "citations": False},
            "technical": {"tone": "precise", "complexity": "high", "citations": True},
            "casual": {"tone": "conversational", "complexity": "low", "citations": False},
            "persuasive": {"tone": "engaging", "complexity": "medium", "citations": True}
        }
        self.default_style = "professional"
        self.max_output_length = 10000
        self.min_output_length = 500
        
        self.logger.info(f"Writer agent initialized: {self.agent_id}")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a writing task.
        
        Args:
            task_data: Task data containing writing requirements
            
        Returns:
            Writing results and content
        """
        start_time = datetime.utcnow()
        task_type = task_data.get("type", "general_writing")
        
        try:
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "started")
            
            # Extract task parameters
            content_data = task_data.get("content", {})
            writing_style = task_data.get("style", self.default_style)
            output_format = task_data.get("format", "markdown")
            target_audience = task_data.get("audience", "general")
            content_length = task_data.get("length", "medium")
            include_citations = task_data.get("citations", False)
            
            if not content_data and task_type not in ["template_generation", "style_guide"]:
                raise AgentException("Content data is required for writing task", error_code="MISSING_CONTENT")
            
            # Validate parameters
            if output_format not in self.supported_formats:
                self.logger.warning(f"Unsupported format {output_format}, defaulting to markdown")
                output_format = "markdown"
            
            self.logger.info(f"Starting writing task: {task_type} in {writing_style} style")
            
            # Store writing task start in memory
            await self.memory.store_conversation(
                message=f"Starting {task_type} writing task in {writing_style} style for {target_audience} audience",
                role="system",
                metadata={
                    "task_type": task_type,
                    "style": writing_style,
                    "format": output_format,
                    "audience": target_audience
                }
            )
            
            # Execute writing task based on type
            if task_type == "report_writing":
                result = await self._write_report(
                    data=content_data,
                    style=writing_style,
                    audience=target_audience,
                    include_citations=include_citations
                )
            elif task_type == "article_writing":
                result = await self._write_article(
                    content=content_data,
                    style=writing_style,
                    length=content_length
                )
            elif task_type == "summary_writing":
                result = await self._write_summary(
                    source_content=content_data,
                    style=writing_style,
                    length=content_length
                )
            elif task_type == "content_editing":
                result = await self._edit_content(
                    content=content_data.get("text", ""),
                    editing_type=task_data.get("editing_type", "comprehensive"),
                    style=writing_style
                )
            elif task_type == "content_structuring":
                result = await self._structure_content(
                    raw_content=content_data,
                    structure_type=task_data.get("structure", "article")
                )
            else:
                # General content creation
                result = await self._create_general_content(
                    content_data=content_data,
                    style=writing_style,
                    format_type=output_format,
                    audience=target_audience
                )
            
            # Format output according to requested format
            formatted_content = await self._format_content(
                content=result.get("content", ""),
                format_type=output_format,
                style=writing_style
            )
            
            result["formatted_content"] = formatted_content
            result["output_format"] = output_format
            
            # Validate content length
            content_length_check = self._validate_content_length(formatted_content)
            result.update(content_length_check)
            
            # Store writing results in memory
            await self.memory.store_insight(
                insight=f"Writing task completed: {result.get('summary', 'Content created successfully')}",
                category="writing_completion",
                confidence=result.get('quality_score', 0.8),
                metadata={
                    "task_type": task_type,
                    "style": writing_style,
                    "word_count": result.get('word_count', 0),
                    "format": output_format,
                    "writing_duration": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            # Add metadata to result
            result.update({
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "task_id": task_data.get("id"),
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": datetime.utcnow().isoformat(),
                "style_used": writing_style,
                "tools_used": [tool.name for tool in self._tools if hasattr(tool, 'name')]
            })
            
            # Log performance metrics
            log_agent_performance(
                self.logger,
                self.agent_id,
                f"writing_{task_type}",
                result["execution_time"],
                success=True,
                metadata={
                    "words_written": result.get('word_count', 0),
                    "quality_score": result.get('quality_score', 0)
                }
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "completed")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store failure in memory
            await self.memory.store_conversation(
                message=f"Writing task failed: {str(e)}",
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
                f"writing_{task_type}",
                execution_time,
                success=False,
                metadata={"error": str(e)}
            )
            
            log_agent_task(self.logger, self.agent_id, task_data.get("id", "unknown"), "failed")
            
            raise
    
    async def _write_report(
        self, 
        data: Dict[str, Any], 
        style: str,
        audience: str,
        include_citations: bool
    ) -> Dict[str, Any]:
        """Write a comprehensive report."""
        self.logger.info("Writing comprehensive report")
        
        # Get content structuring tool
        structuring_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'content_structuring'), None)
        
        if not structuring_tool:
            raise AgentException("Content structuring tool not available", error_code="TOOL_NOT_AVAILABLE")
        
        # Structure the report
        structured_result = await structuring_tool.execute(
            content=str(data),
            structure_type="report"
        )
        
        # Generate report sections
        sections = self._generate_report_sections(data, style, audience)
        
        # Compile full report
        report_content = self._compile_report(sections, style, include_citations)
        
        # Calculate quality metrics
        quality_metrics = self._assess_content_quality(report_content, "report")
        
        return {
            "content": report_content,
            "sections": sections,
            "word_count": len(report_content.split()),
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "style": style,
            "audience": audience,
            "citations_included": include_citations,
            "summary": f"Comprehensive report generated with {len(sections)} sections",
            "content_type": "report"
        }
    
    async def _write_article(
        self, 
        content: Dict[str, Any], 
        style: str,
        length: str
    ) -> Dict[str, Any]:
        """Write an article."""
        self.logger.info(f"Writing {length} {style} article")
        
        # Determine target word count based on length
        target_words = {
            "short": 500,
            "medium": 1500,
            "long": 3000
        }.get(length, 1500)
        
        # Generate article structure
        article_structure = self._create_article_structure(content, style, target_words)
        
        # Write article content
        article_content = self._write_article_content(article_structure, style)
        
        # Quality assessment
        quality_metrics = self._assess_content_quality(article_content, "article")
        
        return {
            "content": article_content,
            "structure": article_structure,
            "word_count": len(article_content.split()),
            "target_word_count": target_words,
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "style": style,
            "length_category": length,
            "summary": f"{style.title()} article written with {len(article_content.split())} words",
            "content_type": "article"
        }
    
    async def _write_summary(
        self, 
        source_content: Dict[str, Any], 
        style: str,
        length: str
    ) -> Dict[str, Any]:
        """Write a summary of provided content."""
        self.logger.info(f"Writing {length} summary in {style} style")
        
        # Extract key information
        key_points = self._extract_key_points(source_content)
        
        # Determine summary length
        target_words = {
            "brief": 150,
            "medium": 300,
            "detailed": 600
        }.get(length, 300)
        
        # Generate summary
        summary_content = self._generate_summary_content(key_points, style, target_words)
        
        # Quality assessment
        quality_metrics = self._assess_content_quality(summary_content, "summary")
        
        return {
            "content": summary_content,
            "key_points": key_points,
            "word_count": len(summary_content.split()),
            "target_word_count": target_words,
            "compression_ratio": len(str(source_content)) / len(summary_content),
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "style": style,
            "summary": f"Summary generated capturing {len(key_points)} key points",
            "content_type": "summary"
        }
    
    async def _edit_content(
        self, 
        content: str, 
        editing_type: str,
        style: str
    ) -> Dict[str, Any]:
        """Edit and improve existing content."""
        self.logger.info(f"Editing content with {editing_type} editing")
        
        original_word_count = len(content.split())
        
        # Perform different types of editing
        if editing_type == "grammar":
            edited_content = self._grammar_check_and_fix(content)
        elif editing_type == "style":
            edited_content = self._style_improvement(content, style)
        elif editing_type == "structure":
            edited_content = self._structural_editing(content)
        else:
            # Comprehensive editing
            edited_content = self._comprehensive_editing(content, style)
        
        # Track changes
        changes_made = self._track_content_changes(content, edited_content)
        
        # Quality assessment
        quality_metrics = self._assess_content_quality(edited_content, "edited_content")
        
        return {
            "content": edited_content,
            "original_content": content,
            "editing_type": editing_type,
            "original_word_count": original_word_count,
            "word_count": len(edited_content.split()),
            "changes_made": changes_made,
            "improvement_score": quality_metrics["improvement_score"],
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "style": style,
            "summary": f"Content edited with {len(changes_made)} improvements made",
            "content_type": "edited_content"
        }
    
    async def _structure_content(
        self, 
        raw_content: Dict[str, Any], 
        structure_type: str
    ) -> Dict[str, Any]:
        """Structure raw content into organized format."""
        self.logger.info(f"Structuring content as {structure_type}")
        
        # Get content structuring tool
        structuring_tool = next((tool for tool in self._tools if hasattr(tool, 'name') and tool.name == 'content_structuring'), None)
        
        if structuring_tool:
            structured_result = await structuring_tool.execute(
                content=str(raw_content),
                structure_type=structure_type
            )
            structured_content = structured_result.get("structured_content", {})
        else:
            # Fallback manual structuring
            structured_content = self._manual_content_structuring(raw_content, structure_type)
        
        # Generate structured output
        formatted_structure = self._format_structured_content(structured_content, structure_type)
        
        # Quality assessment
        quality_metrics = self._assess_content_quality(formatted_structure, "structured_content")
        
        return {
            "content": formatted_structure,
            "structure_type": structure_type,
            "sections": structured_content,
            "word_count": len(formatted_structure.split()),
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "summary": f"Content structured into {structure_type} format with organized sections",
            "content_type": "structured_content"
        }
    
    async def _create_general_content(
        self, 
        content_data: Dict[str, Any], 
        style: str,
        format_type: str,
        audience: str
    ) -> Dict[str, Any]:
        """Create general content based on provided data."""
        self.logger.info(f"Creating general content in {style} style for {audience} audience")
        
        # Extract content elements
        title = content_data.get("title", "Generated Content")
        main_points = content_data.get("points", [])
        context = content_data.get("context", "")
        
        # Generate content structure
        content_outline = self._create_content_outline(title, main_points, context, style)
        
        # Write content based on outline
        generated_content = self._write_from_outline(content_outline, style, audience)
        
        # Quality assessment
        quality_metrics = self._assess_content_quality(generated_content, "general_content")
        
        return {
            "content": generated_content,
            "outline": content_outline,
            "title": title,
            "word_count": len(generated_content.split()),
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "style": style,
            "audience": audience,
            "summary": f"General content created with {len(main_points)} main points",
            "content_type": "general_content"
        }
    
    async def _format_content(
        self, 
        content: str, 
        format_type: str,
        style: str
    ) -> str:
        """Format content according to specified format."""
        if format_type == "markdown":
            return self._format_as_markdown(content, style)
        elif format_type == "html":
            return self._format_as_html(content, style)
        elif format_type == "plain_text":
            return self._format_as_plain_text(content)
        else:
            # Default to markdown
            return self._format_as_markdown(content, style)
    
    def _generate_report_sections(self, data: Dict[str, Any], style: str, audience: str) -> List[Dict[str, str]]:
        """Generate report sections (simplified implementation)."""
        sections = [
            {
                "title": "Executive Summary",
                "content": f"This report provides a comprehensive analysis based on the provided data. Key findings and recommendations are outlined for {audience} stakeholders."
            },
            {
                "title": "Introduction",
                "content": f"The purpose of this report is to analyze and present findings from the available data in a {style} manner suitable for {audience} review."
            },
            {
                "title": "Analysis",
                "content": f"Based on the data analysis, several key patterns and insights have been identified. The {style} approach ensures thorough examination of all relevant factors."
            },
            {
                "title": "Recommendations",
                "content": f"The following recommendations are proposed based on the analysis, tailored for {audience} implementation and consideration."
            },
            {
                "title": "Conclusion",
                "content": "In conclusion, this analysis provides valuable insights that can inform strategic decision-making and future planning initiatives."
            }
        ]
        return sections
    
    def _compile_report(self, sections: List[Dict[str, str]], style: str, include_citations: bool) -> str:
        """Compile report sections into full document."""
        compiled = f"# Research Report\n\n"
        
        for section in sections:
            compiled += f"## {section['title']}\n\n{section['content']}\n\n"
        
        if include_citations:
            compiled += "## References\n\n[1] Data analysis performed by AI Research Assistant\n[2] Multi-source analysis and synthesis\n\n"
        
        return compiled
    
    def _assess_content_quality(self, content: str, content_type: str) -> Dict[str, Any]:
        """Assess the quality of written content."""
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Simple quality metrics (in real implementation, these would be more sophisticated)
        metrics = {
            "readability_score": 0.8,  # Placeholder
            "grammar_score": 0.9,      # Placeholder
            "coherence_score": 0.85,   # Placeholder
            "completeness_score": 0.9, # Placeholder
            "style_consistency": 0.88,  # Placeholder
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": word_count / max(sentence_count, 1)
        }
        
        # Calculate overall score
        quality_scores = [
            metrics["readability_score"],
            metrics["grammar_score"],
            metrics["coherence_score"],
            metrics["completeness_score"],
            metrics["style_consistency"]
        ]
        
        metrics["overall_score"] = sum(quality_scores) / len(quality_scores)
        metrics["improvement_score"] = 0.15  # Placeholder for editing tasks
        
        return metrics
    
    def _validate_content_length(self, content: str) -> Dict[str, Any]:
        """Validate content length against requirements."""
        word_count = len(content.split())
        
        validation = {
            "word_count": word_count,
            "meets_min_length": word_count >= self.min_output_length,
            "within_max_length": word_count <= self.max_output_length,
            "length_status": "appropriate"
        }
        
        if word_count < self.min_output_length:
            validation["length_status"] = "too_short"
        elif word_count > self.max_output_length:
            validation["length_status"] = "too_long"
        
        return validation
    
    def _create_article_structure(self, content: Dict[str, Any], style: str, target_words: int) -> Dict[str, Any]:
        """Create article structure (simplified implementation)."""
        return {
            "introduction": {"target_words": int(target_words * 0.15), "content": "Article introduction"},
            "main_body": {"target_words": int(target_words * 0.70), "content": "Main article content"},
            "conclusion": {"target_words": int(target_words * 0.15), "content": "Article conclusion"}
        }
    
    def _write_article_content(self, structure: Dict[str, Any], style: str) -> str:
        """Write article content from structure."""
        content = "# Article Title\n\n"
        
        for section, details in structure.items():
            content += f"## {section.title()}\n\n{details['content']}\n\n"
        
        return content
    
    def _extract_key_points(self, source_content: Dict[str, Any]) -> List[str]:
        """Extract key points from source content."""
        # Simplified key point extraction
        return [
            "Key insight from source material",
            "Important finding highlighted in analysis", 
            "Strategic recommendation from data review",
            "Critical observation from content examination"
        ]
    
    def _generate_summary_content(self, key_points: List[str], style: str, target_words: int) -> str:
        """Generate summary content from key points."""
        summary = f"# Summary\n\nThis summary captures the essential elements from the source material:\n\n"
        
        for i, point in enumerate(key_points[:5], 1):  # Limit to top 5 points
            summary += f"{i}. {point}\n"
        
        summary += f"\nThis {style} summary provides a comprehensive overview of the key findings and recommendations."
        
        return summary
    
    def _grammar_check_and_fix(self, content: str) -> str:
        """Perform grammar checking and fixes."""
        # Simplified grammar fixes (in real implementation, would use NLP tools)
        fixed_content = content
        fixed_content = re.sub(r'\s+', ' ', fixed_content)  # Fix spacing
        fixed_content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', fixed_content)  # Fix sentence spacing
        return fixed_content
    
    def _style_improvement(self, content: str, target_style: str) -> str:
        """Improve content style."""
        # Simplified style improvement
        if target_style == "professional":
            # Make more formal
            content = content.replace("can't", "cannot")
            content = content.replace("won't", "will not")
        
        return content
    
    def _structural_editing(self, content: str) -> str:
        """Perform structural editing."""
        # Simplified structural improvements
        return content
    
    def _comprehensive_editing(self, content: str, style: str) -> str:
        """Perform comprehensive editing."""
        edited = self._grammar_check_and_fix(content)
        edited = self._style_improvement(edited, style)
        edited = self._structural_editing(edited)
        return edited
    
    def _track_content_changes(self, original: str, edited: str) -> List[str]:
        """Track changes made during editing."""
        # Simplified change tracking
        changes = []
        
        if len(edited.split()) != len(original.split()):
            changes.append("Word count adjusted")
        
        if "cannot" in edited and "can't" in original:
            changes.append("Contractions expanded for formality")
        
        changes.append("Grammar and style improvements applied")
        
        return changes
    
    def _manual_content_structuring(self, raw_content: Dict[str, Any], structure_type: str) -> Dict[str, Any]:
        """Manual content structuring fallback."""
        if structure_type == "article":
            return {
                "title": raw_content.get("title", "Article Title"),
                "introduction": "Introduction content",
                "main_content": str(raw_content),
                "conclusion": "Conclusion content"
            }
        else:
            return {"content": str(raw_content)}
    
    def _format_structured_content(self, structured_content: Dict[str, Any], structure_type: str) -> str:
        """Format structured content."""
        formatted = ""
        
        for section, content in structured_content.items():
            formatted += f"## {section.title()}\n\n{content}\n\n"
        
        return formatted
    
    def _create_content_outline(self, title: str, main_points: List[str], context: str, style: str) -> Dict[str, Any]:
        """Create content outline."""
        return {
            "title": title,
            "introduction": f"Introduction to {title}",
            "main_points": main_points,
            "context": context,
            "style_guide": style
        }
    
    def _write_from_outline(self, outline: Dict[str, Any], style: str, audience: str) -> str:
        """Write content from outline."""
        content = f"# {outline['title']}\n\n"
        content += f"## Introduction\n\n{outline['introduction']}\n\n"
        
        content += "## Main Content\n\n"
        for i, point in enumerate(outline['main_points'], 1):
            content += f"{i}. {point}\n"
        
        if outline['context']:
            content += f"\n## Context\n\n{outline['context']}\n\n"
        
        content += f"\nThis content is written in {style} style for {audience} audience."
        
        return content
    
    def _format_as_markdown(self, content: str, style: str) -> str:
        """Format content as Markdown."""
        # Content is already in markdown format in our implementation
        return content
    
    def _format_as_html(self, content: str, style: str) -> str:
        """Format content as HTML."""
        # Simple markdown to HTML conversion
        html_content = content.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)
        html_content = html_content.replace("## ", "<h2>").replace("\n", "</h2>\n")
        html_content = f"<html><body>\n{html_content}\n</body></html>"
        return html_content
    
    def _format_as_plain_text(self, content: str) -> str:
        """Format content as plain text."""
        # Remove markdown formatting
        plain_text = re.sub(r'#+\s*', '', content)
        plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_text)
        plain_text = re.sub(r'\*(.*?)\*', r'\1', plain_text)
        return plain_text
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools for writer agent."""
        return [
            "content_structuring",
            "grammar_checking",
            "style_optimization",
            "readability_enhancement",
            "citation_formatting",
            "document_formatting"
        ]
    
    async def get_writing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent writing history.
        
        Args:
            limit: Maximum number of writing entries to return
            
        Returns:
            List of writing history entries
        """
        insights = await self.memory.get_recent_insights("writing_completion", limit=limit)
        
        history = []
        for insight in insights:
            history.append({
                "timestamp": insight.timestamp.isoformat(),
                "task_type": insight.metadata.get("task_type", "unknown"),
                "style": insight.metadata.get("style", "unknown"),
                "word_count": insight.metadata.get("word_count", 0),
                "format": insight.metadata.get("format", "unknown"),
                "quality_score": insight.importance,
                "summary": insight.content.get("insight", "")
            })
        
        return history
    
    def get_supported_styles(self) -> Dict[str, Dict[str, Any]]:
        """Get supported writing styles and their characteristics."""
        return self.writing_styles
    
    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return self.supported_formats