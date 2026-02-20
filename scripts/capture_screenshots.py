import os
import time
from playwright.sync_api import sync_playwright

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def capture_screenshots():
    print("Starting Playwright to capture screenshots...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--window-size=1920,1080'])
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501")
        
        # Streamlit dashboards take a moment to load
        print("Waiting for initial load...")
        time.sleep(10)
        
        tabs = [
            ("Fleet Overview", "dashboard_fleet_overview.png"),
            ("Risk Assessment", "dashboard_risk_assessment.png"),
            ("Maintenance Schedule", "dashboard_maintenance_schedule.png"),
            ("Maintenance History", "dashboard_maintenance_history.png"),
            ("Operational Context", "dashboard_operational_context.png"),
        ]
        
        for tab_name, filename in tabs:
            print(f"Switching to tab: {tab_name}")
            
            try:
                # Streamlit renders sidebar radio items inside labels
                page.locator(f"label:has-text('{tab_name}')").first.click()
            except Exception as e:
                print(f"Failed to click {tab_name} using label. Trying exact text...")
                try:
                    page.get_by_text(tab_name, exact=True).first.click()
                except Exception as e2:
                    print(f"Still failed: {e2}")
                
            # Wait for Streamlit to render the new page
            time.sleep(6)
            
            filepath = os.path.join(ASSETS_DIR, filename)
            page.screenshot(path=filepath, full_page=False)
            print(f"--> Saved screenshot to {filepath}")
            
        browser.close()
        print("All screenshots updated!")

if __name__ == "__main__":
    capture_screenshots()
