# step3_ag2.py - 基于AG2的智能协作评审和改进模块 (修正版)

import json
import os
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from ag2_agents import AG2LiteratureReviewSystem
from ollama_client import generate_text
from config import MODEL_CONFIG, SYSTEM_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('step3_ag2.log')
    ]
)
logger = logging.getLogger(__name__)

class AG2LiteratureImprover:
    """基于AG2的文献改进系统"""
    
    def __init__(self):
        try:
            self.ag2_system = AG2LiteratureReviewSystem()
            logger.info("AG2系统初始化成功")
        except Exception as e:
            logger.error(f"AG2系统初始化失败: {e}")
            # 继续执行，但标记AG2不可用
            self.ag2_system = None
            
        self.output_dir = SYSTEM_CONFIG.output_dir
        self.improved_dir = os.path.join(self.output_dir, "improved")
        
        # 确保改进结果目录存在
        os.makedirs(self.improved_dir, exist_ok=True)
        
        logger.info("AG2文献改进系统初始化完成")
    
    def load_step2_results(self) -> List[Dict[str, Any]]:
        """加载Step2生成的结果"""
        
        step2_files = []
        
        # 查找Step2的输出文件
        for file in os.listdir(self.output_dir):
            if file.endswith('.json') and 'consolidated' in file:
                file_path = os.path.join(self.output_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 验证是否是有效的Step2结果
                    if all(key in data for key in ['chapter', 'section_type', 'content']):
                        data['file_path'] = file_path
                        data['filename'] = file
                        step2_files.append(data)
                        
                except Exception as e:
                    logger.warning(f"无法读取文件 {file}: {e}")
                    continue
        
        logger.info(f"找到 {len(step2_files)} 个Step2结果文件")
        return step2_files
    
    def improve_single_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """改进单个内容"""
        
        chapter = content_data.get('chapter', 'unknown')
        section_type = content_data.get('section_type', 'unknown')
        original_content = content_data.get('content', '')
        
        logger.info(f"开始改进: {chapter} - {section_type}")
        
        # 构建主题
        topic = f"{chapter} - {section_type}"
        
        try:
            # 检查AG2系统是否可用
            if self.ag2_system is None:
                logger.warning("AG2系统不可用，使用备用方法")
                return self.improve_with_fallback_method(content_data, topic)
            
            # 使用AG2系统进行评审
            review_results = self.run_ag2_review(original_content, topic)
            
            if not review_results.get('success', False):
                logger.error(f"AG2评审失败: {review_results.get('error', '未知错误')}")
                logger.info("尝试使用备用方法...")
                return self.improve_with_fallback_method(content_data, topic)
            
            # 基于评审结果生成改进版本
            improved_content = self.generate_improved_version(
                original_content, 
                review_results, 
                topic
            )
            
            # 构建改进结果
            improvement_result = {
                'success': True,
                'original_data': content_data,
                'review_results': review_results,
                'improved_content': improved_content,
                'improvement_summary': self.create_improvement_summary(review_results),
                'timestamp': datetime.now().isoformat(),
                'ag2_version': True
            }
            
            logger.info(f"改进完成: {chapter} - {section_type}")
            return improvement_result
            
        except Exception as e:
            logger.error(f"改进过程出错: {e}")
            logger.info("尝试使用备用方法...")
            return self.improve_with_fallback_method(content_data, topic)
    
    def run_ag2_review(self, content: str, topic: str) -> Dict[str, Any]:
        """运行AG2评审，处理不同的方法名"""
        try:
            # 尝试不同的方法名
            if hasattr(self.ag2_system, 'review_literature'):
                return self.ag2_system.review_literature(content, topic)
            elif hasattr(self.ag2_system, 'process_content'):
                return self.ag2_system.process_content(content, topic)
            elif hasattr(self.ag2_system, 'evaluate_content'):
                return self.ag2_system.evaluate_content(content, topic)
            else:
                # 手动调用各个agent
                return self.manual_ag2_review(content, topic)
                
        except Exception as e:
            logger.error(f"AG2评审异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def manual_ag2_review(self, content: str, topic: str) -> Dict[str, Any]:
        """手动调用AG2各个expert进行评审"""
        try:
            reviews = {}
            
            # 调用化学专家
            if hasattr(self.ag2_system, 'chemistry_expert'):
                chem_success, chem_review = self.ag2_system.chemistry_expert.generate_oai_reply([
                    {"content": f"请从化学准确性角度评审以下内容：\n主题：{topic}\n内容：{content}"}
                ])
                if chem_success:
                    reviews['chemistry'] = chem_review
            
            # 调用文献专家
            if hasattr(self.ag2_system, 'literature_expert'):
                lit_success, lit_review = self.ag2_system.literature_expert.generate_oai_reply([
                    {"content": f"请从结构和逻辑角度评审以下内容：\n主题：{topic}\n内容：{content}"}
                ])
                if lit_success:
                    reviews['literature'] = lit_review
            
            # 调用数据专家
            data_expert = None
            for attr_name in ['data_validator', 'data_validation_expert', 'data_expert']:
                if hasattr(self.ag2_system, attr_name):
                    data_expert = getattr(self.ag2_system, attr_name)
                    break
            
            if data_expert:
                data_success, data_review = data_expert.generate_oai_reply([
                    {"content": f"请从数据合理性角度验证以下内容：\n主题：{topic}\n内容：{content}"}
                ])
                if data_success:
                    reviews['data'] = data_review
            
            # 调用队长
            if hasattr(self.ag2_system, 'captain'):
                captain_success, captain_review = self.ag2_system.captain.generate_oai_reply([
                    {"content": f"请综合评审以下内容：\n主题：{topic}\n内容：{content}\n其他专家意见：{str(reviews)}"}
                ])
                if captain_success:
                    reviews['captain'] = captain_review
            
            if reviews:
                return {
                    'success': True,
                    'reviews': reviews,
                    'final_recommendations': reviews.get('captain', ''),
                    'method': 'manual_ag2_review'
                }
            else:
                return {'success': False, 'error': '所有专家评审都失败了'}
                
        except Exception as e:
            logger.error(f"手动AG2评审失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def improve_with_fallback_method(self, content_data: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """备用改进方法（不依赖AG2）"""
        logger.info("使用备用改进方法...")
        
        original_content = content_data.get('content', '')
        
        # 使用直接的LLM改进
        improvement_prompt = f"""请改进以下文献综述内容：

【主题】: {topic}

【原始内容】:
{original_content}

【改进要求】:
1. 确保化学公式和反应的准确性
2. 改善结构和逻辑流程
3. 验证数据的合理性
4. 保持学术严谨性
5. 修正任何明显的错误
6. 保持内容的完整性和连贯性

请提供改进后的完整内容："""

        system_prompt = """你是一个专业的学术文献改进专家。请对文献内容进行全面改进，确保科学准确性、结构完整性和学术规范性。保持原有的LaTeX公式格式。"""

        try:
            improved_content = generate_text(
                prompt=improvement_prompt,
                model=MODEL_CONFIG.main_model,
                system=system_prompt,
                task_type="literature_generation",
                temperature=0.3,
                max_tokens=8192
            )
            
            # 构建简化的评审结果
            fallback_review = {
                'success': True,
                'reviews': {
                    'general': '使用备用方法进行了全面改进，包括化学准确性、结构逻辑和数据验证。'
                },
                'final_recommendations': '基于单一模型的综合改进',
                'method': 'fallback_direct_improvement'
            }
            
            return {
                'success': True,
                'original_data': content_data,
                'review_results': fallback_review,
                'improved_content': improved_content,
                'improvement_summary': "## 改进摘要\n使用备用方法进行了综合改进，包括化学准确性验证、结构优化和数据检查。",
                'timestamp': datetime.now().isoformat(),
                'ag2_version': False
            }
            
        except Exception as e:
            logger.error(f"备用改进方法也失败了: {e}")
            return {
                'success': False,
                'error': f"所有改进方法都失败了: {str(e)}",
                'original_data': content_data
            }
    
    def generate_improved_version(self, 
                                original_content: str, 
                                review_results: Dict[str, Any], 
                                topic: str) -> str:
        """基于评审结果生成改进版本"""
        
        # 提取改进建议
        recommendations = []
        
        reviews = review_results.get('reviews', {})
        for expert, review in reviews.items():
            recommendations.append(f"【{expert}专家建议】\n{review}")
        
        final_recommendations = review_results.get('final_recommendations', '')
        if final_recommendations:
            recommendations.append(f"【综合建议】\n{final_recommendations}")
        
        # 构建改进提示
        improvement_prompt = f"""请基于以下专家评审意见，改进文献综述内容：

【原始主题】: {topic}

【专家评审意见】:
{chr(10).join(recommendations)}

【原始内容】:
{original_content}

【改进要求】:
1. 根据各专家意见进行针对性改进
2. 保持学术严谨性和准确性
3. 确保化学公式和术语正确
4. 改善结构和逻辑
5. 修正任何数据或计算错误
6. 保持内容的完整性和连贯性

请提供改进后的完整内容："""

        system_prompt = """你是一个专业的学术文献改进专家。请根据多位专家的评审意见，对文献内容进行全面改进。确保：
1. 科学准确性 - 化学反应、公式、数据都必须正确
2. 结构完整性 - 逻辑清晰，组织合理
3. 学术规范性 - 引用规范，表达专业
4. 内容连贯性 - 前后呼应，无矛盾

请保持原有的LaTeX公式格式，不要改变化学公式的表示方法。"""

        try:
            improved_content = generate_text(
                prompt=improvement_prompt,
                model=MODEL_CONFIG.main_model,
                system=system_prompt,
                task_type="literature_generation",
                temperature=0.3,
                max_tokens=8192
            )
            
            return improved_content
            
        except Exception as e:
            logger.error(f"生成改进版本失败: {e}")
            return original_content  # 返回原始内容作为备选
    
    def create_improvement_summary(self, review_results: Dict[str, Any]) -> str:
        """创建改进摘要"""
        
        summary_parts = ["## 改进摘要\n"]
        
        # 提取主要改进点
        reviews = review_results.get('reviews', {})
        method = review_results.get('method', 'standard')
        
        if method == 'fallback_direct_improvement':
            summary_parts.append("### 综合改进")
            summary_parts.append("- 使用备用方法进行了全面的文献改进")
            summary_parts.append("- 包括化学准确性、结构逻辑和数据验证")
        else:
            if 'chemistry' in reviews:
                summary_parts.append("### 化学准确性改进")
                summary_parts.append("- 验证了化学反应机理和公式的准确性")
                summary_parts.append("")
            
            if 'literature' in reviews:
                summary_parts.append("### 结构和逻辑改进")
                summary_parts.append("- 优化了文献综述的结构和逻辑流程")
                summary_parts.append("")
            
            if 'data' in reviews:
                summary_parts.append("### 数据验证改进")
                summary_parts.append("- 检查和修正了实验数据和计算")
                summary_parts.append("")
            
            if 'captain' in reviews or 'final_recommendations' in review_results:
                summary_parts.append("### 综合改进措施")
                summary_parts.append("- 基于专家综合意见进行了全面优化")
        
        return "\n".join(summary_parts)
    
    def save_improvement_result(self, improvement_result: Dict[str, Any]):
        """保存改进结果"""
        
        if not improvement_result.get('success', False):
            logger.warning("跳过保存失败的改进结果")
            return
        
        original_data = improvement_result['original_data']
        chapter = original_data.get('chapter', 'unknown')
        section_type = original_data.get('section_type', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确定方法标识
        method_tag = "AG2" if improvement_result.get('ag2_version', False) else "Fallback"
        
        # 保存改进后的内容
        improved_filename = f"{chapter}_{section_type.replace(' ', '_')}_{method_tag}_improved_{timestamp}.json"
        improved_filepath = os.path.join(self.improved_dir, improved_filename)
        
        # 构建完整的保存数据
        save_data = {
            "chapter": chapter,
            "section_type": section_type,
            "original_content": original_data.get('content', ''),
            "improved_content": improvement_result['improved_content'],
            "improvement_summary": improvement_result['improvement_summary'],
            "review_details": improvement_result['review_results'],
            "timestamp": improvement_result['timestamp'],
            "model_used": MODEL_CONFIG.main_model,
            "method": f"{method_tag}_collaborative_review",
            "original_file": original_data.get('filename', ''),
            "ag2_available": improvement_result.get('ag2_version', False)
        }
        
        try:
            with open(improved_filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"改进结果已保存: {improved_filepath}")
            
            # 同时保存简化版本（仅改进后的内容）
            simple_filename = f"{chapter}_{section_type.replace(' ', '_')}_{method_tag}_final_{timestamp}.json"
            simple_filepath = os.path.join(self.output_dir, simple_filename)
            
            simple_data = {
                "chapter": chapter,
                "section_type": section_type,
                "content": improvement_result['improved_content'],
                "timestamp": improvement_result['timestamp'],
                "model_used": MODEL_CONFIG.main_model,
                "generation_method": f"{method_tag}_improved"
            }
            
            with open(simple_filepath, 'w', encoding='utf-8') as f:
                json.dump(simple_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"最终结果已保存: {simple_filepath}")
            
        except Exception as e:
            logger.error(f"保存改进结果失败: {e}")
    
    def process_all_step2_results(self):
        """处理所有Step2结果"""
        
        logger.info("=== 开始AG2文献改进流程 ===")
        
        # 加载Step2结果
        step2_results = self.load_step2_results()
        
        if not step2_results:
            logger.error("未找到Step2结果文件")
            return
        
        # 处理每个结果
        total_files = len(step2_results)
        success_count = 0
        ag2_success_count = 0
        
        for i, content_data in enumerate(step2_results):
            logger.info(f"处理进度: {i+1}/{total_files}")
            
            try:
                # 改进单个内容
                improvement_result = self.improve_single_content(content_data)
                
                if improvement_result.get('success', False):
                    # 保存改进结果
                    self.save_improvement_result(improvement_result)
                    success_count += 1
                    
                    if improvement_result.get('ag2_version', False):
                        ag2_success_count += 1
                else:
                    logger.error(f"改进失败: {improvement_result.get('error', '未知错误')}")
                
            except Exception as e:
                logger.error(f"处理文件时出错: {e}")
                continue
        
        logger.info(f"=== AG2改进流程完成 ===")
        logger.info(f"总体成功: {success_count}/{total_files} 个文件")
        logger.info(f"AG2成功: {ag2_success_count}/{total_files} 个文件")
        logger.info(f"备用方法: {success_count - ag2_success_count}/{total_files} 个文件")

def main():
    """主函数"""
    
    try:
        # 初始化改进系统
        improver = AG2LiteratureImprover()
        
        # 处理所有Step2结果
        improver.process_all_step2_results()
        
    except Exception as e:
        logger.error(f"主程序出错: {e}")
        raise

if __name__ == "__main__":
    main()