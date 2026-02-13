import re
from playwright.sync_api import Page, expect
import pytest
import subprocess
import time
import sys
import os

# This test assumes the app is already running. 
# The pre-commit hook will handle starting/stopping the app.

def test_loan_eligibility_submission(page: Page):
    # 1. Navigate to the app
    # Try localhost:8000
    try:
        page.goto("http://127.0.0.1:8000")
    except Exception:
        pytest.fail("Could not reach http://127.0.0.1:8000. Is the app running?")

    # 2. Verify Title
    expect(page).to_have_title("Loan Eligibility Predictor")

    # 3. Fill out the form
    page.fill("#total_debt", "50000")
    page.fill("#total_assets", "150000")
    page.fill("#net_profit", "20000")
    page.fill("#total_revenue", "100000")
    page.fill("#loan_amount", "25000")
    page.fill("#years_in_business", "5")
    
    # Handle slider and select
    page.evaluate("document.getElementById('credit_score').value = 700")
    page.evaluate("document.getElementById('credit_score').dispatchEvent(new Event('input'))")
    
    page.select_option("#business_type", "Tech")

    # 4. Submit
    page.click(".btn-submit")

    # 5. Verify Result
    # Wait for result card
    result_card = page.locator(".result-card")
    expect(result_card).to_be_visible()
    
    # Check for text
    expect(result_card).to_contain_text("Result: Approved")
    # Optional: Check confidence
    # expect(result_card).to_contain_text("Confidence:")
