import asyncio
from playwright.async_api import async_playwright

async def wake_app(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"Navigating to {url}...")
        
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(3000)
        
        button = page.locator("button:has-text('Yes, get this app back up')")
        
        if await button.is_visible():
            print("Found the button! Waking up app...")
            await button.click()
            await page.wait_for_timeout(5000) 
            print("App successfully woken up!")
        else:
            print("App seems to be already active.")
        
        await browser.close()

async def main():
    # CHANGE THIS URL to your new app's URL
    urls = [
        "https://smart-diffusion-dialysis.streamlit.app/" 
    ]
    for url in urls:
        await wake_app(url)

if __name__ == "__main__":
    asyncio.run(main())
