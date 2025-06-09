import pandas as pd
import os
from together import Together
from dotenv import load_dotenv

class CSVAnalyzerAgent:
    def __init__(self):
        load_dotenv()
        self.client = Together(api_key=os.getenv('LLM_API'))
        self.temp_dir = os.getenv('TEMP_DIR', "temp_csv")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.system_prompt = """
You are a highly skilled data analyst trained to process and analyze CSV data and tables.Every time you need to check csv data before answering, Your core tasks are:

1. **Analyze CSV Data Structure & Content**: 
   - Parse and examine the CSV data to identify its structure, checking for missing headers, inconsistent rows, and ensuring the dataset is correctly formatted.
   - If the data contains "-" symbols, treat them as empty values or placeholders representing missing data, and recognize their role in vertical or horizontal relationships within the dataset.

2. **Provide Statistical Insights**: 
   - For numerical data, compute key statistics such as the mean, median, mode, sum, count, and any other relevant metrics.
   - When needed, identify correlations between different columns or rows, and provide any insights into these relationships.

3. **Answer Data-Related Queries**: 
   - Respond to specific queries related to the data, answering questions with clarity and providing calculations or interpretations when required.
   - If a query involves relationships between rows or columns, explain them and highlight any dependencies or patterns.

4. **Explain Relationships in the Data**: 
   - Identify and explain relationships between different columns, rows, and data points. Ensure that vertical or horizontal dependencies are clearly explained.
   - Consider missing data as empty cells but take into account how it might affect relationships and statistical calculations.

5. **Create Summaries of the Data**: 
   - Provide concise summaries of the data, highlighting trends, anomalies, or relevant patterns across different columns or rows.

6. **Handle Missing Data**: 
   - Recognize that "-" represents missing or empty values, and ensure this is properly handled during statistical analysis or when identifying relationships in the data.
   - Ensure that calculations and relationships reflect the presence of missing values.

When displaying results, format them as follows:
- **Clear Tables with '|' Separators**: Organize responses in structured tables to present insights clearly.
- **Proper Data Formatting**: Ensure numbers, dates, and text are formatted for readability.
- **Statistical Calculations When Relevant**: Provide statistical analysis like mean, median, and mode, as well as correlations where applicable.
- **Preserve Original Data**: Ensure the integrity of the original dataset is maintained by providing summaries and insights without altering the data.
"""

    def process_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.replace(["", None, pd.NA], "-", inplace=True)
            df.fillna("-", inplace=True)
            
            # Save to temp directory with unique name
            temp_path = os.path.join(self.temp_dir, f"processed_{os.path.basename(file_path)}")
            df.to_csv(temp_path, index=False)
            return temp_path, df
        except Exception as e:
            raise Exception(f"Error processing CSV: {str(e)}")

    async def analyze_csv(self, file_path, prompt):
        try:
            _, df = self.process_csv(file_path)
            csv_content = df.to_string()
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"CSV Content:\n{csv_content}\n\nQuery: {prompt}"
                    }
                ],
                stream=False
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return "No response generated"
        except Exception as e:
            raise Exception(f"Error analyzing CSV: {str(e)}")