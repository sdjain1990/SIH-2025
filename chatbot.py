# -*- coding: utf-8 -*-
import torch
import gradio as gr
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import os
import time
import hashlib
import warnings
import gc
from datetime import datetime
import openai
from typing import Dict, Optional, List
import json
import re

# -------------------------------
# Performance Settings
# -------------------------------
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

# Set PyTorch threads based on CPU cores
num_cores = os.cpu_count()
torch.set_num_threads(min(4, num_cores if num_cores else 4))

# Enable cudnn benchmarking for better GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# -------------------------------
# OpenAI Configuration
# -------------------------------
# Set your OpenAI API key here or as environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<key_here>")  # Replace with your actual API key
# openai.api_key = OPENAI_API_KEY

# -------------------------------
# Enhanced Response Enhancement Module 
# -------------------------------
class ResponseEnhancer:
    def __init__(self, api_key: str):
        """Initialize the Response Enhancer with OpenAI API"""
        self.api_key = api_key
        openai.api_key = api_key
        self.enhancement_cache = {}

    def create_enhancement_prompt(self, original_response: str, user_question: str) -> str:
        """Create a detailed prompt for OpenAI to enhance the response"""
        prompt = f"""
You are an expert Ayurvedic health consultant. Create a comprehensive, structured response for the following question.
IMPORTANT: Provide detailed, practical information including specific recipes with ingredients and preparation methods.

User Question: {user_question}

Base Context: {original_response}

Please provide a COMPREHENSIVE response in the following JSON structure. BE VERY DETAILED:
{{
    "summary": "A comprehensive 3-4 line summary of the complete answer",
    "main_response": {{
        "overview": "Detailed explanation of the condition/topic (at least 100 words)",
        "key_points": ["Detailed point 1", "Detailed point 2", "Detailed point 3", "Detailed point 4", "Detailed point 5"]
    }},
    "ayurvedic_approach": {{
        "principles": "Detailed explanation of relevant Ayurvedic principles and concepts",
        "dosha_considerations": {{
            "vata": "Specific considerations for Vata dosha",
            "pitta": "Specific considerations for Pitta dosha",
            "kapha": "Specific considerations for Kapha dosha"
        }},
        "treatments": [
            "Detailed treatment option 1 with explanation",
            "Detailed treatment option 2 with explanation",
            "Detailed treatment option 3 with explanation"
        ]
    }},
    "dietary_recommendations": {{
        "general_guidelines": "Overall dietary principles to follow",
        "foods_to_include": [
            "Food 1: reason why it's beneficial",
            "Food 2: reason why it's beneficial",
            "Food 3: reason why it's beneficial",
            "Food 4: reason why it's beneficial",
            "Food 5: reason why it's beneficial"
        ],
        "foods_to_avoid": [
            "Food 1: reason to avoid",
            "Food 2: reason to avoid",
            "Food 3: reason to avoid"
        ],
        "meal_timing": "Best times to eat and meal frequency",
        "recipes": [
            {{
                "name": "Recipe Name 1",
                "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity"],
                "preparation": "Step by step preparation method",
                "benefits": "Why this recipe is beneficial",
                "best_time": "When to consume"
            }},
            {{
                "name": "Recipe Name 2",
                "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity"],
                "preparation": "Step by step preparation method",
                "benefits": "Why this recipe is beneficial",
                "best_time": "When to consume"
            }},
            {{
                "name": "Recipe Name 3",
                "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity"],
                "preparation": "Step by step preparation method",
                "benefits": "Why this recipe is beneficial",
                "best_time": "When to consume"
            }}
        ]
    }},
    "lifestyle_modifications": [
        "Detailed lifestyle change 1 with implementation tips",
        "Detailed lifestyle change 2 with implementation tips",
        "Detailed lifestyle change 3 with implementation tips",
        "Detailed lifestyle change 4 with implementation tips"
    ],
    "daily_routine": {{
        "morning": "Detailed morning routine with timings",
        "afternoon": "Afternoon practices",
        "evening": "Evening routine",
        "night": "Night time practices for better health"
    }},
    "herbs_and_remedies": {{
        "herbs": [
            {{
                "name": "Herb name 1",
                "benefits": "Specific benefits",
                "dosage": "How much to take",
                "preparation": "How to prepare/consume",
                "precautions": "Any precautions"
            }},
            {{
                "name": "Herb name 2",
                "benefits": "Specific benefits",
                "dosage": "How much to take",
                "preparation": "How to prepare/consume",
                "precautions": "Any precautions"
            }}
        ],
        "home_remedies": [
            {{
                "name": "Remedy 1",
                "ingredients": "What you need",
                "preparation": "How to prepare",
                "usage": "How and when to use",
                "effectiveness": "Expected results"
            }},
            {{
                "name": "Remedy 2",
                "ingredients": "What you need",
                "preparation": "How to prepare",
                "usage": "How and when to use",
                "effectiveness": "Expected results"
            }}
        ]
    }},
    "yoga_and_exercise": [
        "Specific yoga asana 1 with benefits and practice tips",
        "Specific yoga asana 2 with benefits and practice tips",
        "Breathing exercise with technique",
        "Physical activity recommendations"
    ],
    "precautions": [
        "Important precaution 1",
        "Important precaution 2",
        "Important precaution 3"
    ],
    "expected_results": "Timeline and what improvements to expect",
    "when_to_consult": "Specific situations when professional help is needed"
}}

Ensure all sections are filled with detailed, practical, and actionable information. Include specific measurements, timings, and practical tips.
"""
        return prompt

    def enhance_response(self, original_response: str, user_question: str) -> Dict:
        """Enhance the response using OpenAI API"""
        # Check cache first
        cache_key = hashlib.md5(f"{user_question}_{original_response}".encode()).hexdigest()
        if cache_key in self.enhancement_cache:
            return self.enhancement_cache[cache_key]

        try:
            # Create the enhancement prompt
            prompt = self.create_enhancement_prompt(original_response, user_question)

            # Call OpenAI API with increased token limit for detailed response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",  # Using 16k model for longer responses
                messages=[
                    {"role": "system", "content": "You are an expert Ayurvedic health consultant providing detailed, comprehensive health advice with specific recipes, remedies, and practical guidance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000,  # Increased for more detailed responses
                response_format={"type": "json_object"}
            )

            # Parse the response
            enhanced_content = response.choices[0].message.content

            # Try to parse as JSON
            try:
                structured_response = json.loads(enhanced_content)
            except json.JSONDecodeError:
                # Fallback to comprehensive structure if JSON fails
                structured_response = self.create_comprehensive_fallback(original_response, user_question)

            # Cache the enhanced response
            self.enhancement_cache[cache_key] = structured_response

            return structured_response

        except Exception as e:
            print(f"⚠️ OpenAI Enhancement failed: {e}")
            # Return a comprehensive structured version
            return self.create_comprehensive_fallback(original_response, user_question)

    def create_comprehensive_fallback(self, original_response: str, user_question: str) -> Dict:
        """Create a comprehensive structured response when enhancement fails"""
        return {
            "summary": f"Comprehensive Ayurvedic guidance for: {user_question[:100]}. This response provides detailed dietary recommendations, lifestyle modifications, and traditional remedies based on Ayurvedic principles.",
            "main_response": {
                "overview": original_response if original_response else "Based on Ayurvedic principles, this condition requires a holistic approach combining diet, lifestyle, and natural remedies for optimal healing.",
                "key_points": [
                    "Balance your doshas through appropriate diet and lifestyle",
                    "Follow a regular daily routine (Dinacharya) for optimal health",
                    "Include healing herbs and spices in your daily diet",
                    "Practice yoga and meditation for mind-body balance",
                    "Focus on prevention through proper nutrition and lifestyle"
                ]
            },
            "ayurvedic_approach": {
                "principles": "Ayurveda emphasizes treating the root cause rather than symptoms, focusing on balancing the three doshas (Vata, Pitta, Kapha) and strengthening the digestive fire (Agni).",
                "dosha_considerations": {
                    "vata": "For Vata types: Focus on warm, moist, grounding foods. Avoid cold, dry, and raw foods. Maintain regular routine.",
                    "pitta": "For Pitta types: Emphasize cooling, sweet foods. Avoid spicy, sour, and fermented foods. Stay cool and calm.",
                    "kapha": "For Kapha types: Choose light, warm, spicy foods. Avoid heavy, cold, and oily foods. Stay active."
                },
                "treatments": [
                    "Panchakarma detoxification for deep cleansing and rejuvenation",
                    "Herbal medicine tailored to your constitution and condition",
                    "Dietary therapy based on your dosha and digestive capacity"
                ]
            },
            "dietary_recommendations": {
                "general_guidelines": "Eat freshly cooked, warm meals. Avoid processed foods. Eat your largest meal at lunch when digestion is strongest.",
                "foods_to_include": [
                    "Whole grains: Rich in fiber and nutrients for sustained energy",
                    "Fresh seasonal vegetables: Provide essential vitamins and minerals",
                    "Digestive spices (ginger, cumin, coriander): Enhance digestion and metabolism",
                    "Ghee: Nourishes tissues and supports digestion",
                    "Fresh fruits: Best eaten alone, not with meals"
                ],
                "foods_to_avoid": [
                    "Processed and packaged foods: Lack prana (life force)",
                    "Cold beverages: Diminish digestive fire",
                    "Incompatible food combinations: Like milk with sour fruits"
                ],
                "meal_timing": "Breakfast: 7-8 AM (light), Lunch: 12-1 PM (largest meal), Dinner: 6-7 PM (light)",
                "recipes": [
                    {
                        "name": "Healing Kitchari",
                        "ingredients": [
                            "1/2 cup basmati rice",
                            "1/2 cup yellow mung dal",
                            "1 tbsp ghee",
                            "1 tsp cumin seeds",
                            "1 tsp coriander seeds",
                            "1/2 tsp turmeric",
                            "4 cups water",
                            "Salt to taste"
                        ],
                        "preparation": "1. Wash rice and dal. 2. Heat ghee, add spices. 3. Add rice, dal, and water. 4. Cook until soft (30-40 min). 5. Season with salt.",
                        "benefits": "Easy to digest, detoxifying, balances all doshas",
                        "best_time": "Lunch or light dinner"
                    },
                    {
                        "name": "Golden Turmeric Milk",
                        "ingredients": [
                            "1 cup warm milk (dairy or plant-based)",
                            "1/2 tsp turmeric powder",
                            "1/4 tsp ginger powder",
                            "Pinch of black pepper",
                            "1 tsp honey (add when lukewarm)"
                        ],
                        "preparation": "1. Warm milk gently. 2. Add spices and stir. 3. Simmer for 5 minutes. 4. Cool slightly and add honey.",
                        "benefits": "Anti-inflammatory, immunity booster, promotes good sleep",
                        "best_time": "Evening, 1 hour before bed"
                    },
                    {
                        "name": "Digestive Tea",
                        "ingredients": [
                            "1 tsp cumin seeds",
                            "1 tsp coriander seeds",
                            "1 tsp fennel seeds",
                            "2 cups water"
                        ],
                        "preparation": "1. Boil water. 2. Add seeds and simmer 5 minutes. 3. Strain and serve warm.",
                        "benefits": "Enhances digestion, reduces bloating, detoxifying",
                        "best_time": "After meals"
                    }
                ]
            },
            "lifestyle_modifications": [
                "Wake up before sunrise (5:30-6:30 AM) for optimal energy and health",
                "Practice oil pulling with sesame or coconut oil for oral and systemic health",
                "Perform daily self-massage (Abhyanga) with warm oil suitable for your dosha",
                "Maintain regular sleep schedule, sleeping by 10 PM for proper restoration"
            ],
            "daily_routine": {
                "morning": "5:30 AM - Wake up, tongue scraping, oil pulling, warm water with lemon, yoga/exercise, meditation, warm breakfast",
                "afternoon": "12-1 PM - Main meal, short walk, avoid sleeping",
                "evening": "5-6 PM - Light exercise or walk, light dinner by 7 PM",
                "night": "9 PM - Wind down, warm milk, meditation, sleep by 10 PM"
            },
            "herbs_and_remedies": {
                "herbs": [
                    {
                        "name": "Ashwagandha",
                        "benefits": "Reduces stress, improves sleep, boosts immunity",
                        "dosage": "1/2 tsp powder twice daily",
                        "preparation": "Mix with warm milk or water",
                        "precautions": "Avoid in pregnancy, may increase heat in Pitta types"
                    },
                    {
                        "name": "Triphala",
                        "benefits": "Gentle detox, improves digestion, supports eye health",
                        "dosage": "1/2 tsp at bedtime",
                        "preparation": "Soak in warm water, drink before bed",
                        "precautions": "Start with small dose, may cause loose stools initially"
                    }
                ],
                "home_remedies": [
                    {
                        "name": "Ginger-Honey-Lemon Remedy",
                        "ingredients": "Fresh ginger, honey, lemon",
                        "preparation": "Grate ginger, mix with honey and lemon juice",
                        "usage": "1 tsp twice daily before meals",
                        "effectiveness": "Improves digestion and immunity within 1-2 weeks"
                    },
                    {
                        "name": "Fennel Seed Water",
                        "ingredients": "1 tsp fennel seeds, 1 cup water",
                        "preparation": "Soak overnight, strain and drink",
                        "usage": "First thing in morning",
                        "effectiveness": "Reduces bloating, improves metabolism"
                    }
                ]
            },
            "yoga_and_exercise": [
                "Sun Salutations (Surya Namaskar): 5-12 rounds for overall health and vitality",
                "Pranayama (Breathing exercises): Alternate nostril breathing for balance",
                "Meditation: 15-20 minutes daily for mental clarity and stress reduction",
                "Walking: 30 minutes daily, preferably in nature"
            ],
            "precautions": [
                "Always consult a qualified practitioner before starting new treatments",
                "Start with small doses of herbs and gradually increase",
                "Individual results may vary based on constitution and condition"
            ],
            "expected_results": "Initial improvements in 2-4 weeks, significant changes in 2-3 months with consistent practice",
            "when_to_consult": "Seek immediate professional help if symptoms worsen, new symptoms appear, or no improvement after 4 weeks of consistent practice"
        }

    def format_structured_response(self, structured_data: Dict) -> str:
        """Format the structured data into a comprehensive readable markdown format"""
        formatted = f"""
# 🌿 Comprehensive Ayurvedic Health Guidance

## 📋 Executive Summary
{structured_data.get('summary', 'Comprehensive Ayurvedic guidance provided.')}

---

## 🎯 Detailed Analysis

### Overview
{structured_data.get('main_response', {}).get('overview', 'Based on Ayurvedic principles, a holistic approach is recommended.')}

### Key Insights
"""
        key_points = structured_data.get('main_response', {}).get('key_points', [])
        for i, point in enumerate(key_points, 1):
            formatted += f"{i}. {point}\n"

        formatted += """
---

## 🕉️ Ayurvedic Treatment Approach

### Core Principles
"""
        formatted += structured_data.get('ayurvedic_approach', {}).get('principles', 'Based on traditional Ayurvedic wisdom') + "\n\n"

        formatted += "### Dosha-Specific Considerations\n\n"
        dosha_considerations = structured_data.get('ayurvedic_approach', {}).get('dosha_considerations', {})
        if isinstance(dosha_considerations, dict):
            formatted += f"**🔴 Vata Dosha:**\n{dosha_considerations.get('vata', 'Balance with warm, grounding practices')}\n\n"
            formatted += f"**🔥 Pitta Dosha:**\n{dosha_considerations.get('pitta', 'Balance with cooling, calming practices')}\n\n"
            formatted += f"**💧 Kapha Dosha:**\n{dosha_considerations.get('kapha', 'Balance with stimulating, warming practices')}\n\n"
        else:
            formatted += f"{dosha_considerations}\n\n"

        formatted += "### Recommended Treatments\n"
        treatments = structured_data.get('ayurvedic_approach', {}).get('treatments', [])
        for treatment in treatments:
            formatted += f"• {treatment}\n"

        formatted += """
---

## 🥗 Detailed Dietary Guidelines

### General Principles
"""
        formatted += structured_data.get('dietary_recommendations', {}).get('general_guidelines', 'Follow Ayurvedic dietary principles') + "\n\n"

        formatted += "### ✅ Foods to Include\n"
        foods_include = structured_data.get('dietary_recommendations', {}).get('foods_to_include', [])
        for food in foods_include:
            formatted += f"• **{food}**\n"

        formatted += "\n### ❌ Foods to Avoid\n"
        foods_avoid = structured_data.get('dietary_recommendations', {}).get('foods_to_avoid', [])
        for food in foods_avoid:
            formatted += f"• {food}\n"

        formatted += f"\n### ⏰ Meal Timing\n{structured_data.get('dietary_recommendations', {}).get('meal_timing', 'Eat at regular times')}\n"

        formatted += "\n---\n\n## 🍲 Ayurvedic Recipes\n\n"
        recipes = structured_data.get('dietary_recommendations', {}).get('recipes', [])
        for i, recipe in enumerate(recipes, 1):
            if isinstance(recipe, dict):
                formatted += f"### Recipe {i}: {recipe.get('name', 'Traditional Recipe')}\n\n"
                formatted += "**Ingredients:**\n"
                for ingredient in recipe.get('ingredients', []):
                    formatted += f"• {ingredient}\n"
                formatted += f"\n**Preparation:**\n{recipe.get('preparation', 'Follow traditional method')}\n\n"
                formatted += f"**Benefits:** {recipe.get('benefits', 'Promotes health and balance')}\n"
                formatted += f"**Best Time to Consume:** {recipe.get('best_time', 'As recommended')}\n\n"
            else:
                formatted += f"### Recipe {i}\n{recipe}\n\n"

        formatted += """
---

## 🧘 Lifestyle Modifications
"""
        lifestyle = structured_data.get('lifestyle_modifications', [])
        for i, mod in enumerate(lifestyle, 1):
            formatted += f"{i}. {mod}\n"

        formatted += "\n---\n\n## 📅 Ideal Daily Routine (Dinacharya)\n\n"
        daily_routine = structured_data.get('daily_routine', {})
        if isinstance(daily_routine, dict):
            formatted += f"**🌅 Morning:** {daily_routine.get('morning', 'Start day with healthy practices')}\n\n"
            formatted += f"**☀️ Afternoon:** {daily_routine.get('afternoon', 'Maintain energy and focus')}\n\n"
            formatted += f"**🌆 Evening:** {daily_routine.get('evening', 'Wind down gradually')}\n\n"
            formatted += f"**🌙 Night:** {daily_routine.get('night', 'Prepare for restful sleep')}\n\n"

        formatted += """
---

## 🌿 Herbs and Natural Remedies

### Recommended Herbs
"""
        herbs = structured_data.get('herbs_and_remedies', {}).get('herbs', [])
        for herb in herbs:
            if isinstance(herb, dict):
                formatted += f"\n**🌱 {herb.get('name', 'Herb')}**\n"
                formatted += f"• *Benefits:* {herb.get('benefits', 'Multiple health benefits')}\n"
                formatted += f"• *Dosage:* {herb.get('dosage', 'As recommended')}\n"
                formatted += f"• *Preparation:* {herb.get('preparation', 'Traditional preparation')}\n"
                formatted += f"• *Precautions:* {herb.get('precautions', 'Use with care')}\n"
            else:
                formatted += f"• {herb}\n"

        formatted += "\n### Home Remedies\n"
        remedies = structured_data.get('herbs_and_remedies', {}).get('home_remedies', [])
        for remedy in remedies:
            if isinstance(remedy, dict):
                formatted += f"\n**🏠 {remedy.get('name', 'Home Remedy')}**\n"
                formatted += f"• *Ingredients:* {remedy.get('ingredients', 'Natural ingredients')}\n"
                formatted += f"• *Preparation:* {remedy.get('preparation', 'Simple preparation')}\n"
                formatted += f"• *Usage:* {remedy.get('usage', 'As needed')}\n"
                formatted += f"• *Effectiveness:* {remedy.get('effectiveness', 'Proven results')}\n"
            else:
                formatted += f"• {remedy}\n"

        formatted += """
---

## 🧘‍♀️ Yoga and Exercise Recommendations
"""
        yoga = structured_data.get('yoga_and_exercise', [])
        for practice in yoga:
            formatted += f"• {practice}\n"

        formatted += """
---

## ⚠️ Important Precautions
"""
        precautions = structured_data.get('precautions', [])
        for precaution in precautions:
            formatted += f"⚡ {precaution}\n"

        formatted += f"""
---

## 📈 Expected Results
{structured_data.get('expected_results', 'Results vary based on individual constitution and adherence to recommendations')}

## 🏥 When to Seek Professional Help
{structured_data.get('when_to_consult', 'Consult a healthcare professional if symptoms persist or worsen')}

---

*💡 This comprehensive guidance combines traditional Ayurvedic wisdom with practical modern applications. For best results, follow recommendations consistently and make gradual changes to your lifestyle.*

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return formatted

# -------------------------------
# Enhanced Optimized Chatbot Class
# -------------------------------
class EnhancedMedicalChatbot:
    def __init__(self, model_path=r"C:\Users\amano\Downloads\enhanced_medical_model-20250921T135657Z-1-001\enhanced_medical_model",
                 use_openai_enhancement=True):
        print("="*60)
        print("🚀 INITIALIZING ENHANCED MEDICAL CHATBOT")
        print("="*60)

        start_time = time.time()

        # Initialize OpenAI enhancer if enabled
        self.use_openai_enhancement = use_openai_enhancement
        if use_openai_enhancement and OPENAI_API_KEY != "your-api-key-here":
            try:
                self.enhancer = ResponseEnhancer(OPENAI_API_KEY)
                print("✅ OpenAI Response Enhancement: ENABLED")
            except Exception as e:
                print(f"⚠️ OpenAI Enhancement initialization failed: {e}")
                self.use_openai_enhancement = False
                self.enhancer = None
        else:
            print("ℹ️ OpenAI Response Enhancement: DISABLED")
            self.enhancer = None

        # Device configuration with detailed info
        self.setup_device()

        # Load model and tokenizer
        print(f"\n📁 Loading model from: {model_path}")
        try:
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)
            print("✅ Tokenizer loaded successfully")

            # Load model with memory optimization
            print("📊 Loading model weights...")
            if torch.cuda.is_available():
                self.model = OpenAIGPTLMHeadModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = OpenAIGPTLMHeadModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True
                )
            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Optimize model for inference
        self.optimize_model()

        # Initialize response cache
        self.response_cache = {}
        self.cache_hits = 0
        self.total_requests = 0

        # Set special tokens if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_time = time.time() - start_time
        print(f"\n✅ Model initialization complete in {load_time:.2f} seconds")
        print("="*60)

        # Warm up the model
        self.warmup_model()

    def setup_device(self):
        """Configure and display device information"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n🖥️ SYSTEM CONFIGURATION:")
        print("-"*40)

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Detected: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.2f} GB")
            print(f"🔧 CUDA Version: {torch.version.cuda}")

            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("⚠️ No GPU detected - Using CPU")
            print("💡 TIP: Install CUDA and GPU version of PyTorch for 10x faster responses")
            print(f"🔧 CPU Cores: {os.cpu_count()}")
            print(f"🔧 Threads allocated: {torch.get_num_threads()}")

    def optimize_model(self):
        """Apply model optimizations for faster inference"""
        print("\n⚡ Applying optimizations...")

        # Move model to device
        self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        # Apply optimization based on device
        if torch.cuda.is_available():
            try:
                self.model = self.model.half()
                print("✅ Using FP16 precision for faster GPU inference")
            except Exception as e:
                print(f"⚠️ FP16 not available, using FP32: {e}")
                self.model = self.model.float()

            # Enable TF32 for Ampere GPUs
            if hasattr(torch.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ TF32 enabled for Ampere GPUs")
        else:
            try:
                torch.set_flush_denormal(True)
            except:
                pass

    def warmup_model(self):
        """Warm up the model with a dummy input"""
        print("\n🔥 Warming up model...")
        try:
            dummy_input = "Hello"
            _ = self.generate_response_internal(dummy_input, max_length=10)
            self.response_cache.clear()
            print("✅ Model warmed up and ready")
        except Exception as e:
            print(f"⚠️ Warmup failed (non-critical): {e}")

    def get_cache_key(self, prompt, max_length, use_enhancement):
        """Generate a unique cache key for the prompt"""
        cache_string = f"{prompt}_{max_length}_{use_enhancement}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    @torch.no_grad()
    def generate_response_internal(self, prompt, max_length=200):
        """Internal method for generating responses with optimizations"""
        formatted_prompt = f"Question: {prompt}\nAnswer:"

        # Tokenize with optimized settings
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        ).to(self.device)

        # Optimized generation parameters
        generation_config = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_length,
            "min_length": 10,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_beams": 1,
            "use_cache": True
        }

        # Generate response
        outputs = self.model.generate(**generation_config)

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response

    def generate_response(self, prompt, max_length=200, enhance_with_openai=None):
        """Generate response with optional OpenAI enhancement - returns ONLY enhanced version when enabled"""
        self.total_requests += 1

        # Determine if we should enhance this response
        should_enhance = enhance_with_openai if enhance_with_openai is not None else self.use_openai_enhancement

        # Check cache first
        cache_key = self.get_cache_key(prompt, max_length, should_enhance)
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cache_rate = (self.cache_hits / self.total_requests) * 100
            print(f"📌 Cache hit! (Rate: {cache_rate:.1f}%)")
            return self.response_cache[cache_key]

        # Generate new response with timing
        start_time = time.time()
        print(f"\n🤖 Generating response for: '{prompt[:50]}...'")

        try:
            if should_enhance and self.enhancer:
                # When enhancement is enabled, generate base response but only return enhanced version
                print("🚀 Generating enhanced response with OpenAI...")
                
                # Generate minimal base response for context
                base_response = self.generate_response_internal(prompt, max_length=100)
                
                # Get enhanced response
                try:
                    structured_response = self.enhancer.enhance_response(base_response, prompt)
                    final_response = self.enhancer.format_structured_response(structured_response)
                    print("✅ Response enhanced successfully")
                except Exception as e:
                    print(f"⚠️ Enhancement failed: {e}")
                    # Fallback to comprehensive structured response
                    structured_response = self.enhancer.create_comprehensive_fallback(base_response, prompt)
                    final_response = self.enhancer.format_structured_response(structured_response)
            else:
                # Only generate base response when enhancement is disabled
                final_response = self.generate_response_internal(prompt, max_length)

            # Cache the response
            if len(self.response_cache) > 100:
                self.response_cache.clear()
                print("🔄 Cache cleared (size limit reached)")

            self.response_cache[cache_key] = final_response

            elapsed_time = time.time() - start_time
            print(f"✅ Response generated in {elapsed_time:.2f}s")

            if elapsed_time > 5:
                print("⚠️ Slow response detected. Consider:")
                print("   - Using GPU (if not already)")
                print("   - Reducing max_length")
                print("   - Disabling OpenAI enhancement for faster responses")

            return final_response

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."

    def get_stats(self):
        """Get performance statistics"""
        cache_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_rate": cache_rate,
            "cache_size": len(self.response_cache),
            "device": str(self.device),
            "openai_enhancement": "Enabled" if self.use_openai_enhancement else "Disabled"
        }

# -------------------------------
# Enhanced Gradio UI with OpenAI Toggle
# -------------------------------
def create_enhanced_ui(chatbot):
    def respond(message, history, use_enhancement):
        if not message:
            return history, "", ""

        # Show loading status
        start_time = time.time()
        bot_response = chatbot.generate_response(message, enhance_with_openai=use_enhancement)
        response_time = time.time() - start_time

        history.append([message, bot_response])

        # Update stats
        stats = chatbot.get_stats()
        stats_text = f"""
        📊 **Performance Stats:**
        - Device: {stats['device']}
        - Response Time: {response_time:.2f}s
        - Total Requests: {stats['total_requests']}
        - Cache Hit Rate: {stats['cache_rate']:.1f}%
        - Cache Size: {stats['cache_size']}
        - OpenAI Enhancement: {'✅ Active' if use_enhancement else '❌ Disabled'}
        """

        return history, stats_text, ""

    def clear_chat():
        return [], "", ""

    def clear_cache():
        chatbot.response_cache.clear()
        chatbot.cache_hits = 0
        chatbot.total_requests = 0
        return "✅ Cache cleared successfully!"

    # Custom CSS
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    #main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        max-width: 1600px;
        margin: auto;
    }
    #chatbot {
        height: 700px;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        background: white;
        overflow-y: auto;
    }
    #title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 10px;
    }
    .examples-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
    button {
        transition: all 0.3s ease;
    }
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    #stats-box {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        font-family: monospace;
    }
    .enhancement-toggle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="🌿 Enhanced Ayurvedic Health Assistant") as demo:
        with gr.Column(elem_id="main-container"):
            gr.HTML(
                """
                <div id="title">
                    🌿 AI-Enhanced Ayurvedic Health & Nutrition Assistant
                </div>
                <p style="text-align: center; color: #666; font-size: 1.1em;">
                    Comprehensive Health Guidance with Detailed Recipes, Remedies & Lifestyle Plans
                </p>
                <hr style="border: none; height: 2px; background: linear-gradient(to right, #667eea, #764ba2); margin: 20px 0;">
                """
            )

            # Performance and Enhancement status
            device_info = "🚀 GPU Accelerated" if torch.cuda.is_available() else "💻 CPU Mode"
            enhancement_status = "✅ Available" if chatbot.use_openai_enhancement else "❌ Not Configured"

            gr.HTML(
                f"""
                <div style="text-align: center; padding: 10px; background: {'#4CAF50' if torch.cuda.is_available() else '#FF9800'};
                            color: white; border-radius: 10px; margin-bottom: 20px;">
                    <strong>{device_info}</strong> |
                    Model Status: ✅ Ready |
                    OpenAI Enhancement: {enhancement_status}
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_ui = gr.Chatbot(
                        label="💬 Chat with Your AI-Enhanced Ayurvedic Assistant",
                        elem_id="chatbot",
                        height=700,
                        bubble_full_width=False,
                        avatar_images=["🧑", "🌿"]
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Ask your question",
                            placeholder="Example: What is the Ayurvedic treatment for diabetes with diet plan and recipes?",
                            lines=2,
                            scale=4
                        )

                    # OpenAI Enhancement Toggle with explanation
                    with gr.Row():
                        use_enhancement = gr.Checkbox(
                            label="🚀 Enable OpenAI Enhancement (Get detailed recipes, remedies, and structured guidance)",
                            value=chatbot.use_openai_enhancement,
                            interactive=True
                        )

                    with gr.Row():
                        submit = gr.Button("🚀 Send", variant="primary", scale=1)
                        clear = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                        clear_cache_btn = gr.Button("🔄 Clear Cache", variant="secondary", scale=1)

                    # Performance stats display
                    stats_display = gr.Markdown(
                        elem_id="stats-box",
                        value="📊 **Performance Stats:** Waiting for first request..."
                    )
                    cache_status = gr.Textbox(label="Cache Status", visible=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Quick Topics\n\nClick any topic for comprehensive guidance:")

                    # Organized example buttons
                    with gr.Column():
                        gr.Markdown("**🏥 Health Conditions**")
                        example_buttons_health = [
                            ("🩺 Diabetes", "I have diabetes. Provide complete Ayurvedic treatment with specific diet recipes, herbs, daily routine, and lifestyle changes"),
                            ("🫀 Hypertension", "Give me detailed Ayurvedic management for high blood pressure including recipes, herbs, yoga, and daily routine"),
                            ("🦴 Joint Pain", "Provide comprehensive Ayurvedic treatment for arthritis with anti-inflammatory recipes, herbs, oils, and exercises"),
                            ("🦋 Thyroid", "Complete Ayurvedic approach for thyroid disorders with metabolism-boosting recipes and remedies"),
                            ("🔥 Acidity", "Detailed plan for chronic acidity with cooling recipes, herbs, and lifestyle modifications"),
                        ]

                        for btn_text, btn_prompt in example_buttons_health:
                            btn = gr.Button(btn_text, size="sm")
                            btn.click(lambda p=btn_prompt: p, outputs=msg)

                    with gr.Column():
                        gr.Markdown("**🥗 Diet & Nutrition**")
                        example_buttons_diet = [
                            ("⚖️ Weight Loss", "Create comprehensive Ayurvedic weight loss plan with daily meal recipes, metabolism-boosting drinks, and exercise routine"),
                            ("🧘 Find Dosha", "Analyze my dosha and provide personalized diet plan with specific recipes for breakfast, lunch, and dinner"),
                            ("🌅 Morning Routine", "Design complete Ayurvedic morning routine with recipes for morning drinks, breakfast, and practices"),
                            ("☀️ Summer Diet", "Provide cooling summer diet plan with specific recipes, drinks, and foods to balance Pitta"),
                            ("❄️ Winter Diet", "Create warming winter nutrition plan with Kapha-balancing recipes and immunity-boosting foods"),
                        ]

                        for btn_text, btn_prompt in example_buttons_diet:
                            btn = gr.Button(btn_text, size="sm")
                            btn.click(lambda p=btn_prompt: p, outputs=msg)

                    with gr.Column():
                        gr.Markdown("**🌿 Wellness & Remedies**")
                        example_buttons_wellness = [
                            ("💤 Better Sleep", "Provide Ayurvedic protocol for insomnia with bedtime recipes, herbs, and relaxation techniques"),
                            ("🧘 Stress Relief", "Complete stress management plan with calming recipes, herbs, breathing exercises, and daily routine"),
                            ("🛡️ Immunity", "Build immunity with specific recipes, immunity-boosting drinks, herbs, and daily practices"),
                            ("🔥 Digestion", "Improve digestion with digestive recipes, spice combinations, and meal timing guidelines"),
                            ("🌱 Detox", "Complete Ayurvedic detox program with cleansing recipes, drinks, and purification practices"),
                        ]

                        for btn_text, btn_prompt in example_buttons_wellness:
                            btn = gr.Button(btn_text, size="sm")
                            btn.click(lambda p=btn_prompt: p, outputs=msg)

                    gr.Markdown(
                        f"""
                        ### 🎯 What You'll Get

                        **With Enhancement ON:**
                        ✅ Detailed recipes with ingredients
                        ✅ Step-by-step preparation
                        ✅ Dosha-specific recommendations
                        ✅ Daily routine schedules
                        ✅ Herb dosages & preparations
                        ✅ Home remedies
                        ✅ Yoga & exercise plans
                        ✅ Expected results timeline

                        **Without Enhancement:**
                        • Basic response
                        • General guidance
                        • Faster generation

                        ---

                        **⚡ Performance:** {device_info}
                        **🤖 Enhancement:** {enhancement_status}
                        """
                    )

            # Event handlers
            submit.click(
                respond,
                [msg, chatbot_ui, use_enhancement],
                [chatbot_ui, stats_display, msg]
            )
            msg.submit(
                respond,
                [msg, chatbot_ui, use_enhancement],
                [chatbot_ui, stats_display, msg]
            )
            clear.click(clear_chat, outputs=[chatbot_ui, stats_display, msg])
            clear_cache_btn.click(clear_cache, outputs=[cache_status])

            gr.HTML(
                """
                <div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
                    <p style="color: #666;">
                        <strong>⚠️ Disclaimer:</strong> This AI assistant provides Ayurvedic information for educational purposes only.
                        Always consult qualified healthcare practitioners for medical advice and treatment.
                    </p>
                    <p style="color: #999; font-size: 0.9em; margin-top: 10px;">
                        Enhanced with OpenAI for Comprehensive Health Guidance |
                        <span id="timestamp"></span>
                    </p>
                </div>
                <script>
                    document.getElementById('timestamp').innerHTML = new Date().toLocaleString();
                </script>
                """
            )

    return demo

# -------------------------------
# Main Execution with Error Handling
# -------------------------------
def main():
    try:
        # Check for OpenAI API key
        if OPENAI_API_KEY == "your-api-key-here":
            print("⚠️ WARNING: OpenAI API key not configured!")
            print("To enable comprehensive response enhancement with recipes and detailed guidance:")
            print("1. Set OPENAI_API_KEY environment variable, or")
            print("2. Replace 'your-api-key-here' in the code with your actual API key")
            print("")
            response = input("Continue without OpenAI enhancement? (y/n): ")
            if response.lower() != 'y':
                return

        # Model path
        model_path = r"C:\Users\amano\Downloads\enhanced_medical_model-20250921T135657Z-1-001\enhanced_medical_model"

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print("Please check the path and try again.")
            return

        # Initialize chatbot with optimizations
        print("\n🚀 Starting AI-Enhanced Medical Chatbot with Comprehensive Guidance...\n")
        use_openai = OPENAI_API_KEY != "your-api-key-here"
        chatbot = EnhancedMedicalChatbot(model_path, use_openai_enhancement=use_openai)

        # Create and launch UI
        print("\n🌐 Launching Enhanced Gradio Interface...")
        demo = create_enhanced_ui(chatbot)

        # Launch with optimized settings
        demo.queue(max_size=10)
        demo.launch(
            share=True,
            server_name="127.0.0.1",
            server_port=7865,
            inbrowser=True,
            quiet=False,
            show_error=True,
            max_threads=40
        )

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()