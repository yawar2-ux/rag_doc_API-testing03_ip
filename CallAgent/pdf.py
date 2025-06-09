from fpdf import FPDF
from wordcloud import WordCloud
import os
from datetime import datetime

class CallReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Call Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class PDFGenerator:
    @staticmethod
    def generate_call_report(mobile_no, conversation_history, call_summary, sale_opportunity, chat_history, wordcloud_path, avg_sentiment):
        pdf = CallReportPDF()
        pdf.add_page()

        # Mobile No Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Mobile Number', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 6, f"Mobile No: {mobile_no}", 0, 1)
        pdf.ln(5)

        # Call Summary Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Call Summary', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, call_summary or "No summary available")
        pdf.ln(5)

        # Chat History Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Conversation History', 0, 1)
        pdf.set_font('Arial', '', 11)

        for entry in conversation_history:
            speaker = entry.get('type', '').title()
            message = entry.get('message', '')
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, f"{speaker}:", 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, message)
            pdf.ln(2)

        # Sales Opportunity Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Sales Opportunity', 0, 1)
        pdf.set_font('Arial', '', 11)
        if sale_opportunity:
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Opportunities:', 0, 1)
            pdf.set_font('Arial', '', 11)
            for opportunity in sale_opportunity.get('opportunities', []):
                pdf.cell(0, 6, f"- {opportunity}", 0, 1)
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Pain Points:', 0, 1)
            pdf.set_font('Arial', '', 11)
            for pain_point in sale_opportunity.get('pain_points', []):
                pdf.cell(0, 6, f"- {pain_point}", 0, 1)
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Buying Signals:', 0, 1)
            pdf.set_font('Arial', '', 11)
            for signal in sale_opportunity.get('buying_signals', []):
                pdf.cell(0, 6, f"- {signal}", 0, 1)
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Next Steps:', 0, 1)
            pdf.set_font('Arial', '', 11)
            for step in sale_opportunity.get('next_steps', []):
                pdf.cell(0, 6, f"- {step}", 0, 1)
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Probability:', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 6, sale_opportunity.get('probability', 'N/A'), 0, 1)
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Explanation:', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, sale_opportunity.get('explanation', 'N/A'))
        else:
            pdf.cell(0, 6, "No sales opportunities detected.", 0, 1)
        pdf.ln(5)

        # Overall Average Sentiment Score Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Overall Average Sentiment Score', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 6, f"Average Sentiment Score: {avg_sentiment}", 0, 1)
        pdf.ln(5)

        # Wordcloud Section
        if wordcloud_path:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Wordcloud', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.image(wordcloud_path, x=10, y=None, w=100)
            pdf.ln(5)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"call_report_{timestamp}.pdf"

        # Save to reports directory
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        filepath = os.path.join(reports_dir, filename)

        pdf.output(filepath)
        return filepath