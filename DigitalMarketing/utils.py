import json
import ast
import logging
import traceback
from typing import Dict
import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from groq import AsyncGroq
import os


groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['.csv']
TOKEN_OPTIONS = [100, 500, 1000, 2000, 4000]

# Logger
logger = logging.getLogger(__name__)


# Template styles
template_style = {
    "Premium High-Spenders": {
        "bg_color": "#000000",
        "text_color": "#ffffff",
        "accent_color": "#ffd700",
        "button_style": "background: #ffd700; color: #000000;",
        "header_bg": "linear-gradient(135deg, #000000, #1a1a1a)"
    },
    "High-Income Conservatives": {
        "bg_color": "#2c3e50",
        "text_color": "#ecf0f1",
        "accent_color": "#3498db",
        "button_style": "background: #3498db; color: #ffffff;",
        "header_bg": "linear-gradient(135deg, #2c3e50, #34495e)"
    },
    "Digital Enthusiasts": {
        "bg_color": "#ffffff",
        "text_color": "#333333",
        "accent_color": "#2ecc71",
        "button_style": "background: #2ecc71; color: #ffffff;",
        "header_bg": "linear-gradient(135deg, #2ecc71, #27ae60)"
    },
    "Budget Conscious": {
        "bg_color": "#f8f9fa",
        "text_color": "#2c3e50",
        "accent_color": "#3498db",
        "button_style": "background: #3498db; color: #ffffff;",
        "header_bg": "#f8f9fa"
    },
    "Regular Customers": {
        "bg_color": "#ffffff",
        "text_color": "#444444",
        "accent_color": "#e74c3c",
        "button_style": "background: #e74c3c; color: #ffffff;",
        "header_bg": "#ffffff"
    }
}


def validate_file(file: UploadFile) -> None:
    """Validate file size and extension"""
    if not any(file.filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed"
        )
    
    file_size = len(file.file.read())
    file.file.seek(0)  # Reset file pointer
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
        )


def perform_segmentation(df):
    """Perform customer segmentation using K-means clustering"""
    try:
        # Convert string lists to actual lists
        df['preferred_categories'] = df['preferred_categories'].apply(ast.literal_eval)
        
        # Create feature matrix
        features = [
            'monthly_income', 'monthly_spending', 'avg_order_value', 'annual_orders',
            'total_annual_spend', 'email_open_rate', 'visit_frequency_30d', 
            'avg_session_duration_minutes', 'age'
        ]
        
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create segment label mapping
        segment_mapping = {
            0: "Premium High-Spenders",
            1: "High-Income Conservatives",
            2: "Digital Enthusiasts",
            3: "Budget Conscious",
            4: "Regular Customers"
        }
        
        # Add cluster labels
        df['cluster'] = clusters
        df['segment_label'] = pd.Series(clusters).map(segment_mapping)
        
        # Create cluster profiles
        cluster_profiles = []
        for cluster in range(5):
            cluster_data = df[df['cluster'] == cluster]
            profile = {
                'cluster': cluster,
                'segment_label': segment_mapping[cluster],
                'size': len(cluster_data),
                'avg_income': cluster_data['monthly_income'].mean(),
                'avg_spending': cluster_data['monthly_spending'].mean(),
                'avg_age': cluster_data['age'].mean(),
                'avg_order_value': cluster_data['avg_order_value'].mean()
            }
            cluster_profiles.append(profile)
        
        return df, pd.DataFrame(cluster_profiles)
    except Exception as e:
        logger.error(f"Error in perform_segmentation: {str(e)}")
        raise

def generate_chart_data(df):
    """Generate data for interactive charts"""
    try:
        # 1. Income vs Spending Data
        income_spending_data = df.apply(
            lambda x: {
                "customerID": x['customer_id'],
                "monthlyIncome": x['monthly_income'],
                "monthlySpending": x['monthly_spending'],
                "segment": x['segment_label']
            },
            axis=1
        ).tolist()

        # 2. Disposable Income by Segment
        disposable_income_data = df.groupby('segment_label').agg({
            'monthly_income': 'mean',
            'monthly_spending': 'mean'
        }).reset_index().apply(
            lambda x: {
                "segment": x['segment_label'],
                "disposableIncome": x['monthly_income'] - x['monthly_spending']
            },
            axis=1
        ).tolist()

        # 3. Spending Ratio Distribution
        spending_ratio_data = df.groupby('segment_label').agg({
            'monthly_spending': 'sum',
            'monthly_income': 'sum'
        }).reset_index().apply(
            lambda x: {
                "segment": x['segment_label'],
                "spendingRatio": (x['monthly_spending'] / x['monthly_income'] * 100)
            },
            axis=1
        ).tolist()

        # 4. Average Order Value
        avg_order_data = df.groupby('segment_label')['avg_order_value'].mean().reset_index().apply(
            lambda x: {
                "segment": x['segment_label'],
                "averageOrderValue": x['avg_order_value']
            },
            axis=1
        ).tolist()

        # 5. Engagement Matrix
        engagement_data = df.apply(
            lambda x: {
                "customerID": x['customer_id'],
                "emailOpenRate": x['email_open_rate'],
                "visitFrequency": x['visit_frequency_30d'],
                "annualOrders": x['annual_orders'],
                "segment": x['segment_label']
            },
            axis=1
        ).tolist()

        # 6. Category Preferences Heat Map
        categories = sorted(set([cat for cats in df['preferred_categories'] for cat in cats]))
        segment_categories = []
        
        for segment in df['segment_label'].unique():
            segment_df = df[df['segment_label'] == segment]
            total_customers = len(segment_df)
            
            category_percentages = {}
            for cats in segment_df['preferred_categories']:
                for cat in cats:
                    category_percentages[cat] = category_percentages.get(cat, 0) + 1
                    
            row_data = {
                "segment": segment,
                **{cat: (category_percentages.get(cat, 0) / total_customers * 100) for cat in categories}
            }
            segment_categories.append(row_data)

        return {
            "incomeVsSpending": income_spending_data,
            "disposableIncome": disposable_income_data,
            "spendingRatio": spending_ratio_data,
            "averageOrderValue": avg_order_data,
            "engagementMatrix": engagement_data,
            "categoryPreferences": segment_categories
        }
    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        raise

def get_segment_products(recommendations_df, segment):
    """Get unique products for a segment"""
    try:
        segment_recs = recommendations_df[recommendations_df['segment_label'] == segment]
        
        products_by_category = {}
        for _, row in segment_recs.drop_duplicates(subset=['product_id']).iterrows():
            category = row['recommended_category'] if 'recommended_category' in row else row['category']
            if category not in products_by_category:
                products_by_category[category] = []
            
            products_by_category[category].append({
                'name': row['product_name'],
                'price': f"₹{row['price']:.2f}",
                'rating': f"{row['rating']:.1f}/5",
                'score': f"{row['recommendation_score']:.2f}" if 'recommendation_score' in row else "N/A"
            })
        
        return products_by_category
    except Exception as e:
        logger.error(f"Error in get_segment_products: {str(e)}")
        raise

def generate_recommendations(segmented_customers, products_df, purchases_df):
    """Generate segment-aware cross-sell and up-sell recommendations"""
    try:
        segment_strategies = {
            "Premium High-Spenders": {
                "upsell_price_factor": 0.5,
                "min_rating": 4.5,
                "categories": ["premium_electronics", "fine_dining", 'home_decor']
            },
            "High-Income Conservatives": {
                "upsell_price_factor": 0.2,
                "min_rating": 4.2,
                "categories": ["electronics", "appliances", "home_decor"]
            },
            "Digital Enthusiasts": {
                "upsell_price_factor": 0.3,
                "min_rating": 4.0,
                "categories": ["electronics", "premium_electronics", "sports"]
            },
            "Budget Conscious": {
                "upsell_price_factor": 0.1,
                "min_rating": 4.0,
                "categories": ["budget_fashion", "groceries", "home_decor"]
            },
            "Regular Customers": {
                "upsell_price_factor": 0.2,
                "min_rating": 4.0,
                "categories": ["fashion", "beauty", "home_decor"]
            }
        }
        
        cross_sell_recs = []
        upsell_recs = []
        
        for _, customer in segmented_customers.iterrows():
            segment = customer['segment_label']
            strategy = segment_strategies[segment]
            
            # Cross-sell recommendations
            customer_categories = set(customer['preferred_categories'])
            recommended_categories = set(strategy['categories']) - customer_categories
            
            for category in recommended_categories:
                category_products = products_df[
                    (products_df['category'] == category) &
                    (products_df['rating'] >= strategy['min_rating'])
                ].sort_values('rating', ascending=False).head(2)
                
                for _, product in category_products.iterrows():
                    cross_sell_recs.append({
                        'customer_id': customer['customer_id'],
                        'segment_label': segment,
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'recommended_category': category,
                        'price': product['price'],
                        'rating': product['rating'],
                        'recommendation_type': 'cross_sell',
                        'recommendation_reason': f'Popular {category} for {segment} customers'
                    })
            
            # Up-sell recommendations
            customer_purchases = purchases_df[purchases_df['customer_id'] == customer['customer_id']]
            
            if not customer_purchases.empty:
                avg_category_prices = customer_purchases.merge(products_df, on='product_id')\
                    .groupby('category')['price'].mean().to_dict()
                
                for category in customer_categories:
                    if category in avg_category_prices:
                        avg_purchase_price = avg_category_prices[category]
                        max_recommended_price = avg_purchase_price * (1 + strategy['upsell_price_factor'])
                        
                        premium_products = products_df[
                            (products_df['category'] == category) &
                            (products_df['price'] > avg_purchase_price) &  
                            (products_df['price'] <= max_recommended_price) &
                            (products_df['rating'] >= strategy['min_rating'])
                        ].sort_values('price', ascending=False).head(2)
                        
                        for _, product in premium_products.iterrows():
                            price_increase = ((product['price'] - avg_purchase_price) / avg_purchase_price) * 100
                            upsell_recs.append({
                                'customer_id': customer['customer_id'],
                                'segment_label': segment,
                                'product_id': product['product_id'],
                                'product_name': product['product_name'],
                                'category': category,
                                'price': product['price'],
                                'rating': product['rating'],
                                'current_avg_order': avg_purchase_price,
                                'price_increase_percentage': price_increase,
                                'recommendation_type': 'upsell',
                                'recommendation_reason': f'Premium upgrade in {category} for {segment} customers'
                            })
        
        cross_sell_df = pd.DataFrame(cross_sell_recs)
        upsell_df = pd.DataFrame(upsell_recs)
        
        if not cross_sell_df.empty:
            cross_sell_df['recommendation_score'] = (
                cross_sell_df['rating'] * 0.6 +
                (cross_sell_df.groupby('customer_id')['rating'].rank(ascending=False) / 
                 cross_sell_df.groupby('customer_id')['rating'].transform('count')).fillna(1) * 0.4
            ).round(2)
        
        if not upsell_df.empty:
            upsell_df['recommendation_score'] = (
                upsell_df['rating'] * 0.4 +
                (1 - (upsell_df['price_increase_percentage'] / 100).clip(0, 1)) * 0.3 +
                (upsell_df.groupby('customer_id')['rating'].rank(ascending=False) / 
                 upsell_df.groupby('customer_id')['rating'].transform('count')).fillna(1) * 0.3
            ).round(2)
        
        return cross_sell_df, upsell_df
    
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {str(e)}")
        raise

def generate_personalized_template(
    customer_id: str, 
    segment: str, 
    products_by_category: Dict, 
    recommendation_type: str,
    email_content: Dict
) -> str:
    """Generate enhanced personalized HTML email template with modern design"""
    try:
        style = template_style.get(segment, template_style["Regular Customers"])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{email_content['subject_line']}</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Poppins', sans-serif;
                    background-color: #f8f9fa;
                    color: {style['text_color']};
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 600px;
                    margin: 20px auto;
                    background-color: white;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                }}
                
                .header {{
                    background: {style['header_bg']};
                    padding: 40px 20px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: linear-gradient(135deg, {style['accent_color']}22, transparent 70%);
                    z-index: 1;
                }}
                
                .header h1 {{
                    position: relative;
                    z-index: 2;
                    font-size: 32px;
                    font-weight: 700;
                    margin-bottom: 10px;
                    letter-spacing: -0.5px;
                }}
                
                .customer-badge {{
                    display: inline-block;
                    padding: 8px 16px;
                    background: {style['accent_color']}22;
                    border-radius: 50px;
                    font-size: 14px;
                    color: {style['accent_color']};
                    margin-top: 10px;
                }}
                
                .hero-banner {{
                    position: relative;
                    color: white;
                    text-align: center;
                    padding: 60px 30px;
                    background: linear-gradient(45deg, {style['bg_color']}, {style['accent_color']});
                    clip-path: polygon(0 0, 100% 0, 100% 85%, 0 100%);
                }}
                
                .hero-banner p {{
                    font-size: 18px;
                    max-width: 500px;
                    margin: 0 auto;
                    line-height: 1.8;
                }}
                
                .category-section {{
                    padding: 40px 20px;
                    background: white;
                }}
                
                .category-header {{
                    text-align: center;
                    margin-bottom: 30px;
                    position: relative;
                }}
                
                .category-header::after {{
                    content: '';
                    display: block;
                    width: 60px;
                    height: 3px;
                    background: {style['accent_color']};
                    margin: 15px auto;
                    border-radius: 2px;
                }}
                
                .product-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 25px;
                    padding: 0 15px;
                }}
                
                .product-card {{
                    background: white;
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 3px 15px rgba(0,0,0,0.05);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    border: 1px solid #f0f0f0;
                }}
                
                .product-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }}
                
                .product-image {{
                    width: 100%;
                    height: 200px;
                    object-fit: cover;
                    border-radius: 10px;
                    margin-bottom: 15px;
                }}
                
                .product-name {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #2d3436;
                    margin: 10px 0;
                    height: 48px;
                    overflow: hidden;
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                }}
                
                .product-price {{
                    color: {style['accent_color']};
                    font-size: 24px;
                    font-weight: 700;
                    margin: 15px 0;
                }}
                
                .product-rating {{
                    display: inline-flex;
                    align-items: center;
                    padding: 5px 10px;
                    background: {style['accent_color']}11;
                    border-radius: 20px;
                    color: {style['accent_color']};
                    font-size: 14px;
                    margin: 10px 0;
                }}
                
                .cta-button {{
                    display: inline-block;
                    padding: 12px 30px;
                    {style['button_style']}
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: 600;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                    margin: 15px 0;
                }}
                
                .cta-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px {style['accent_color']}44;
                }}
                
                .footer {{
                    background: {style['bg_color']};
                    color: {style['text_color']};
                    padding: 50px 30px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .footer::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 6px;
                    background: linear-gradient(to right, 
                        {style['accent_color']}, 
                        {style['bg_color']}, 
                        {style['accent_color']}
                    );
                }}
                
                .social-links {{
                    margin: 25px 0;
                }}
                
                .social-links a {{
                    display: inline-block;
                    width: 40px;
                    height: 40px;
                    line-height: 40px;
                    border-radius: 50%;
                    background: {style['accent_color']}22;
                    color: {style['text_color']};
                    margin: 0 10px;
                    transition: all 0.3s ease;
                }}
                
                .social-links a:hover {{
                    background: {style['accent_color']};
                    transform: translateY(-3px);
                }}
                
                .footer-links {{
                    margin: 20px 0;
                    font-size: 14px;
                }}
                
                .footer-links a {{
                    color: {style['text_color']};
                    margin: 0 15px;
                    text-decoration: none;
                    position: relative;
                }}
                
                .footer-links a::after {{
                    content: '';
                    position: absolute;
                    bottom: -2px;
                    left: 0;
                    width: 0;
                    height: 2px;
                    background: {style['accent_color']};
                    transition: width 0.3s ease;
                }}
                
                .footer-links a:hover::after {{
                    width: 100%;
                }}
                
                @media (max-width: 600px) {{
                    .product-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .header h1 {{
                        font-size: 24px;
                    }}
                    
                    .hero-banner p {{
                        font-size: 16px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>{email_content['main_heading']}</h1>
                    <div class="customer-badge">ID: {customer_id}</div>
                </div>

                <!-- Hero Banner -->
                <div class="hero-banner">
                    <p>{email_content['intro_paragraph']}</p>
                </div>
        """
        
        # Products by Category
        for category, products in products_by_category.items():
            category_description = email_content['category_descriptions'].get(
                category,
                f"Explore our selection in {category.replace('_', ' ').title()}"
            )
            
            html += f"""
                <div class="category-section">
                    <div class="category-header">
                        <h3>{category.replace('_', ' ').title()}</h3>
                        <p>{category_description}</p>
                    </div>
                    <div class="product-grid">
            """
            
            for product in products:
                html += f"""
                        <div class="product-card">
                            <img src="/api/placeholder/200/200" alt="{product['name']}" class="product-image">
                            <h4 class="product-name">{product['name']}</h4>
                            <div class="product-price">{product['price']}</div>
                            <div class="product-rating">★ {product['rating']}</div>
                            <a href="#" class="cta-button">Shop Now</a>
                        </div>
                """
            
            html += """
                    </div>
                </div>
            """
        
        # Add conclusion and footer
        html += f"""
                <div class="hero-banner" style="margin-top: 40px;">
                    <p>{email_content['conclusion']}</p>
                </div>
                
                <!-- Footer -->
                <div class="footer">
                    <h3>Thank You for Being a Valued Customer</h3>
                    
                    <div class="social-links">
                        <a href="#"><span>fb</span></a>
                        <a href="#"><span>tw</span></a>
                        <a href="#"><span>ig</span></a>
                        <a href="#"><span>in</span></a>
                    </div>
                    
                    <div class="footer-links">
                        <a href="#">View All Products</a>
                        <a href="#">Your Account</a>
                        <a href="#">Support</a>
                    </div>
                    
                    <p style="font-size: 12px; margin-top: 30px;">
                        This email was personally curated for {customer_id}.<br>
                        <a href="#" style="color: {style['text_color']};">Unsubscribe</a> | 
                        <a href="#" style="color: {style['text_color']};">Privacy Policy</a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    except Exception as e:
        logger.error(f"Error in generate_personalized_template: {str(e)}")
        raise

async def generate_email_content(
    customer_id: str, 
    segment: str, 
    products_by_category: Dict, 
    recommendation_type: str,
    max_tokens: int = 1000
) -> Dict:
    """Generate personalized email content using Groq"""
    try:
        logger.info(f"Generating content for customer: {customer_id}, segment: {segment}")
        
        product_info = "\n".join([
            f"Category: {category}\nProducts:\n" + 
            "\n".join([f"- {p['name']} (Price: {p['price']}, Rating: {p['rating']})" 
                      for p in products])
            for category, products in products_by_category.items()
        ])
        
        prompt = f"""As an expert email marketing copywriter, generate personalized email content in JSON format for:
Customer ID: {customer_id}
Segment: {segment}
Recommendation Type: {recommendation_type}
Available Products:
{product_info}

Required JSON structure:
{{
    "subject_line": "Engaging email subject line",
    "main_heading": "Main email heading",
    "intro_paragraph": "Personalized introduction",
    "category_descriptions": {{
        "category_name": "compelling description"
    }},
    "conclusion": "Call to action paragraph"
}}

Tone guidelines for {segment}:
- Premium High-Spenders: Luxurious, exclusive
- Digital Enthusiasts: Tech-savvy, innovative
- Budget Conscious: Value-focused, practical
- Regular Customers: Friendly, balanced
- High-Income Conservatives: Quality-focused, sophisticated

Return only valid JSON."""

        logger.info("Sending prompt to Groq")
        
        completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert email marketing copywriter. Respond only with the requested JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        content = completion.choices[0].message.content
        logger.info("Received response from Groq")

        try:
            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed_content = json.loads(content)
            logger.info("Successfully parsed JSON response")
            return parsed_content
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw content: {content}")
            raise

    except Exception as e:
        logger.error(f"Error in generate_email_content: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback content
        return {
            "subject_line": f"Special {recommendation_type.replace('_', ' ')} recommendations for you",
            "main_heading": f"Personalized {recommendation_type.replace('_', ' ')} Recommendations",
            "intro_paragraph": f"Based on your shopping preferences, we've selected these items just for you.",
            "category_descriptions": {cat: f"Discover our selection in {cat}" for cat in products_by_category.keys()},
            "conclusion": "Shop now and explore these carefully selected items!"
        }


