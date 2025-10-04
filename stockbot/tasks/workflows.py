from __future__ import annotations

import io
import json
import os
from datetime import datetime
from textwrap import dedent
from typing import Any

from crewai import Task
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")


class CompanyDataOutput(BaseModel):
    ticker: str
    company_name: str
    company_info: str


class MarkdownReportCreationTasks:
    async def company_lookup_task(self, agent: Any, company_name: str) -> Task:
        """Build a CrewAI task that resolves a ticker and company summary."""
        description = dedent(
            f"""
            Locate the publicly traded company matching the provided name.

            Company Name: {company_name}

            Requirements:
            - Return the official ticker traded on major exchanges (NYSE, NASDAQ, AMEX, etc.).
            - Provide the full company name that corresponds to that ticker.
            - Summarize key business information (industry, products/services, notable facts).

            Output Schema:
              ticker: str
              company_name: str
              company_info: str
            """
        )

        return Task(
            description=description,
            agent=agent,
            expected_output="Json output with given schema",
            output_pydantic=CompanyDataOutput,
        )


async def update_portfolio(image_attachment: Any) -> str:
    """Convert a portfolio screenshot into structured JSON and persist it."""
    file_bytes = io.BytesIO(await image_attachment.read())
    pil_image = Image.open(file_bytes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_filename = f"temp_portfolio_{timestamp}.png"
    pil_image.save(local_filename)

    try:
        sample_file = genai.upload_file(path=local_filename, display_name="portfolio")
        genai.get_file(name=sample_file.name)

        with open("portfolio.json", "r", encoding="utf-8") as handle:
            portfolio_schema = json.load(handle)

        prompt = dedent(
            f"""
            Read the uploaded image and transcribe the tabular portfolio data into JSON
            matching the provided schema exactly.

            Target schema (from portfolio.json): {portfolio_schema}

            Output ONLY valid JSON: the response must start with '{{' and end with '}}'.
            """
        )

        response = _GEMINI_MODEL.generate_content([sample_file, prompt])
        cleaned_text = response.text.strip()

        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        try:
            portfolio_data = json.loads(cleaned_text.strip())
        except json.JSONDecodeError:
            return "Error: Response was not valid JSON format"

        if not isinstance(portfolio_data, dict):
            return "Error: Response was valid JSON but not a dictionary"

        with open("portfolio.json", "w", encoding="utf-8") as handle:
            json.dump(portfolio_data, handle, indent=4)

        return "Portfolio successfully updated!"

    finally:
        if os.path.exists(local_filename):
            os.remove(local_filename)
