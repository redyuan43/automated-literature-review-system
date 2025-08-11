# ag2_agents.py - 基于AG2的智能Agent协作系统

import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from config import OLLAMA_CONFIG, MODEL_CONFIG

from ollama_client import generate_text, chat_with_model
from config import MODEL_CONFIG, SYSTEM_CONFIG
from chemical_formula_processor import ChemicalFormulaProcessor

logger = logging.getLogger(__name__)

class OllamaLLMConfig:
    """Ollama LLM配置适配器"""
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_CONFIG.small_model
        self.config_list = [{
            "model": MODEL_CONFIG.main_model,  # 使用config.py中的模型
            "base_url": f"{OLLAMA_CONFIG.base_url}/v1",  # 使用config.py中的服务器地址
            "api_key": "ollama",
        }]
    
    def to_dict(self):
        return {"config_list": self.config_list}

class OllamaAgent(ConversableAgent):
    """支持Ollama的Agent基类"""
    
    def __init__(self, name: str, system_message: str, model: str = None, **kwargs):
        self.ollama_model = model or MODEL_CONFIG.small_model
        
        # 设置LLM配置
        llm_config = OllamaLLMConfig(self.ollama_model).to_dict()
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
    
    def generate_oai_reply(self, messages, sender=None, config=None):
        """重写生成回复的方法，使用Ollama"""
        try:
            # 获取最后一条用户消息
            if not messages:
                return False, "没有收到消息"
            
            last_message = messages[-1]
            if isinstance(last_message, dict):
                user_content = last_message.get("content", "")
            else:
                user_content = str(last_message)
            
            # 使用Ollama生成回复
            response = generate_text(
                prompt=user_content,
                model=self.ollama_model,
                system=self.system_message,
                task_type="agent_review",
                temperature=0.5,
                max_tokens=8192  # 增加到8K，充分利用模型能力
            )
            
            return True, response
            
        except Exception as e:
            logger.error(f"Agent {self.name} 生成回复失败: {e}")
            return False, f"生成回复时出错: {str(e)}"

class ChemistryExpertAgent(OllamaAgent):
    """化学专家Agent"""
    
    def __init__(self, model: str = None):
        system_message = """你是一位化学化工领域的资深专家，具有以下专业能力：

核心职责：
1. 验证化学反应机理的科学性和准确性
2. 检查分子结构描述和化学公式的正确性
3. 评估实验方法的合理性和可行性
4. 识别化学术语使用的准确性

评审标准：
- 化学反应机理是否符合基本原理
- 化学公式和分子式是否正确
- 实验条件是否合理可行
- 专业术语使用是否准确

输出格式：
请以"【化学专家评审】"开头，提供具体的评审意见和改进建议。
重点关注化学准确性，如有问题请明确指出并给出正确表述。"""

        super().__init__(
            name="Chemistry_Expert",
            system_message=system_message,
            model=model
        )

class LiteratureReviewAgent(OllamaAgent):
    """文献综述专家Agent"""
    
    def __init__(self, model: str = None):
        system_message = """你是学术文献综述专家，专注于文献综述的结构和逻辑：

核心职责：
1. 确保综述逻辑连贯，结构合理
2. 检查引用的完整性和准确性
3. 优化章节组织和内容流程
4. 提升学术写作质量

评审标准：
- 逻辑结构是否清晰合理
- 内容组织是否有序
- 引用是否规范完整
- 语言表达是否学术化
- 是否有遗漏的重要观点

输出格式：
请以"【文献综述专家评审】"开头，重点关注综述的结构完整性和逻辑连贯性。
提供具体的结构优化建议。"""

        super().__init__(
            name="Literature_Expert", 
            system_message=system_message,
            model=model
        )

class DataValidationAgent(OllamaAgent):
    """数据验证专家Agent"""
    
    def __init__(self, model: str = None):
        system_message = """你是实验数据验证专家，负责验证数据的准确性和合理性：

核心职责：
1. 检查数值数据的合理性
2. 验证单位换算和计算
3. 识别数据中的异常或错误
4. 确保实验数据的科学性

评审标准：
- 数值范围是否合理
- 单位使用是否正确
- 计算过程是否准确
- 数据是否存在明显错误
- 实验参数是否现实可行

输出格式：
请以"【数据验证专家评审】"开头，重点检查所有数值数据。
如发现问题，请明确指出错误位置并提供正确值。"""

        super().__init__(
            name="Data_Validator",
            system_message=system_message,
            model=model
        )

class CaptainAgent(OllamaAgent):
    """队长Agent - 协调和决策"""
    
    def __init__(self, model: str = None):
        system_message = """你是团队协调者，负责综合各专家意见并做出最终决策：

核心职责：
1. 综合各专家的评审意见
2. 协调不同意见的分歧
3. 制定最终的改进方案
4. 确保改进的全面性和一致性

工作流程：
1. 收集所有专家的评审意见
2. 识别共同问题和分歧点
3. 制定综合改进方案
4. 优先处理关键问题

输出格式：
请以"【队长总结】"开头，提供：
- 问题总结（按重要性排序）
- 具体改进建议
- 修改优先级
最后以"【最终改进方案】"结尾。"""

        super().__init__(
            name="Captain",
            system_message=system_message,
            model=model or MODEL_CONFIG.main_model  # 队长使用主模型
        )

class AG2LiteratureReviewSystem:
    """AG2文献综述协作系统"""
    
    def __init__(self):
        self.formula_processor = ChemicalFormulaProcessor()
        
        # 初始化所有Agent
        self.chemistry_expert = ChemistryExpertAgent()
        self.literature_expert = LiteratureReviewAgent()
        self.data_validator = DataValidationAgent()
        self.captain = CaptainAgent()
        
        # 设置GroupChat
        self.setup_group_chat()
        
        logger.info("AG2文献综述系统初始化完成")
    
    def setup_group_chat(self):
        """设置群聊协作"""
        
        # 定义Agent列表
        self.agents = [
            self.chemistry_expert,
            self.literature_expert, 
            self.data_validator,
            self.captain
        ]
        
        # 创建GroupChat
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=SYSTEM_CONFIG.max_agent_rounds,
            speaker_selection_method="round_robin",  # 轮流发言
            allow_repeat_speaker=False
        )
        
        # 创建GroupChatManager - 不使用LLM配置，让每个agent使用自己的配置
        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=None  # 让每个agent使用自己的Ollama配置
        )
    
    def review_literature(self, content: str, topic: str) -> Dict[str, Any]:
        """
        对文献内容进行多Agent协作评审
        
        Args:
            content: 待评审的文献内容
            topic: 文献主题
            
        Returns:
            评审结果和改进建议
        """
        
        logger.info(f"开始AG2协作评审，主题: {topic}")
        
        # 保护化学公式
        protected_content = self.formula_processor.preserve_chemical_content(content)
        
        # 构建初始消息
        initial_message = f"""
请各位专家对以下文献综述内容进行评审：

【主题】: {topic}

【内容】:
{protected_content}  # 使用完整内容，256K上下文足够处理

请每位专家从自己的专业角度进行评审，重点关注：
- 化学专家：化学准确性
- 文献专家：结构和逻辑
- 数据专家：数据合理性
- 队长：综合协调

请开始评审。
"""
        
        try:
            # 开始群聊
            self.group_chat.reset()
            
            # 发起讨论
            chat_result = self.chat_manager.initiate_chat(
                recipient=self.chemistry_expert,  # 从化学专家开始
                message=initial_message,
                max_turns=SYSTEM_CONFIG.max_agent_rounds
            )
            
            # 提取评审结果
            review_results = self._extract_review_results(chat_result)
            
            # 恢复化学公式
            if "improved_content" in review_results:
                review_results["improved_content"] = self.formula_processor.restore_chemical_content(
                    review_results["improved_content"]
                )
            
            logger.info("AG2协作评审完成")
            return review_results
            
        except Exception as e:
            logger.error(f"AG2评审过程出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_content": content
            }
    
    def _extract_review_results(self, chat_result) -> Dict[str, Any]:
        """从聊天结果中提取评审结果"""
        
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "reviews": {},
            "final_recommendations": "",
            "improved_content": ""
        }
        
        # 解析聊天历史
        if hasattr(chat_result, 'chat_history'):
            messages = chat_result.chat_history
        else:
            messages = []
        
        for message in messages:
            if isinstance(message, dict):
                sender = message.get("name", "unknown")
                content = message.get("content", "")
                
                # 提取各专家的评审意见
                if sender == "Chemistry_Expert" and "【化学专家评审】" in content:
                    results["reviews"]["chemistry"] = content
                elif sender == "Literature_Expert" and "【文献综述专家评审】" in content:
                    results["reviews"]["literature"] = content
                elif sender == "Data_Validator" and "【数据验证专家评审】" in content:
                    results["reviews"]["data"] = content
                elif sender == "Captain" and "【队长总结】" in content:
                    results["final_recommendations"] = content
                    
                    # 提取最终改进方案
                    if "【最终改进方案】" in content:
                        improved_section = content.split("【最终改进方案】")[-1].strip()
                        results["improved_content"] = improved_section
        
        return results
    
    def generate_improvement_plan(self, review_results: Dict[str, Any]) -> str:
        """基于评审结果生成改进计划"""
        
        if not review_results.get("success", False):
            return "评审过程中出现错误，无法生成改进计划。"
        
        plan_parts = ["## 文献综述改进计划\n"]
        
        # 添加各专家意见总结
        reviews = review_results.get("reviews", {})
        
        if "chemistry" in reviews:
            plan_parts.append("### 化学专业角度")
            plan_parts.append(reviews["chemistry"])
            plan_parts.append("")
        
        if "literature" in reviews:
            plan_parts.append("### 文献结构角度")
            plan_parts.append(reviews["literature"])
            plan_parts.append("")
        
        if "data" in reviews:
            plan_parts.append("### 数据验证角度")
            plan_parts.append(reviews["data"])
            plan_parts.append("")
        
        # 添加最终建议
        if review_results.get("final_recommendations"):
            plan_parts.append("### 综合改进建议")
            plan_parts.append(review_results["final_recommendations"])
        
        return "\n".join(plan_parts)

def test_ag2_system():
    """测试AG2系统"""
    
    print("=== AG2智能协作系统测试 ===\n")
    
    # 创建测试内容
    test_content = """
# Diels-Alder反应的催化机理研究

## 摘要
本研究探讨了Lewis酸催化的Diels-Alder反应机理。实验发现AlCl₃能够显著提高反应速率。

## 实验部分
我们使用了1,3-丁二烯和马来酸酐作为反应物：
C₄H₆ + C₄H₂O₃ → C₈H₈O₃

反应条件：温度80°C，时间6小时，催化剂AlCl₃ (0.1 mol%)。

## 结果
产率达到95%，这比文献报道的无催化剂条件(65%)有显著提升。
反应活化能从120 kJ/mol降低到85 kJ/mol。

## 机理讨论
Lewis酸通过配位到双烯体的羰基上，降低了LUMO能级，促进了反应进行。
    """
    
    try:
        # 初始化系统
        ag2_system = AG2LiteratureReviewSystem()
        
        # 进行评审
        print("开始多Agent协作评审...")
        results = ag2_system.review_literature(test_content, "Diels-Alder反应催化机理")
        
        # 显示结果
        print("评审完成！\n")
        
        if results.get("success"):
            print("=== 评审结果 ===")
            
            reviews = results.get("reviews", {})
            for expert, review in reviews.items():
                print(f"\n【{expert}专家意见】")
                print(review[:300] + "..." if len(review) > 300 else review)
            
            if results.get("final_recommendations"):
                print(f"\n【最终建议】")
                print(results["final_recommendations"][:300] + "...")
            
            # 生成改进计划
            improvement_plan = ag2_system.generate_improvement_plan(results)
            print(f"\n=== 改进计划 ===")
            print(improvement_plan[:500] + "...")
            
        else:
            print(f"评审失败: {results.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"测试过程中出错: {e}")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_ag2_system()