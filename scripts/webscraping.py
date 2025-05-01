# scripts/webscraping.py

from playwright.sync_api import sync_playwright
import csv

def scrape_bodacc(limit=10, output_path="data/entreprises.csv"):
    url = "https://www.bodacc.fr/pages/annonces-commerciales/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        print("🔄 Chargement de la page BODACC...")
        page.goto(url)

        print("⏳ Attente des résultats...")
        try:
            page.wait_for_selector("article.search-result", timeout=20000)
        except:
            print("⚠️ Aucun résultat détecté après attente.")
            browser.close()
            return

        cards = page.query_selector_all("article.search-result")
        if not cards:
            print("❌ Aucun élément <article.search-result> trouvé.")
            browser.close()
            return

        data = []
        for card in cards[:limit]:
            try:
                titre = card.query_selector("h3").inner_text().strip()
                details = card.query_selector("p").inner_text().strip()
                data.append({"titre": titre, "details": details})
            except:
                continue

        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["titre", "details"])
            writer.writeheader()
            writer.writerows(data)

        print(f"✅ {len(data)} annonces scrappées et enregistrées dans {output_path}")
        browser.close()

if __name__ == "__main__":
    scrape_bodacc(limit=10)
