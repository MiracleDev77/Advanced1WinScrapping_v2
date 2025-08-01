from playwright.sync_api import sync_playwright
import time, random, json, logging, statistics
from datetime import datetime, date
from package import *  # Conserve les fonctions métiers
import numpy as np  # Correction pour l'erreur 'np'

# Configuration du logging
logging.basicConfig(
    filename="scraping.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_data(datas, filename="data.json"):
    """Sauvegarde des données brutes"""
    with open(filename, "a") as f:
        f.write(json.dumps(datas) + "\n")

def get_history_data(frame):
    """Récupère robustement les 50 derniers scores historiques"""
    try:
        # Attendre que l'historique soit chargé avec timeout augmenté
        frame.wait_for_selector('[data-testid^="history-item-"]', timeout=15000)
        
        history_items = frame.query_selector_all(
            '[data-testid^="history-item-"]:not([data-testid="history-item-undefined"])'
        )
        
        scores = []
        for item in history_items:
            text = item.inner_text().strip()
            if text.endswith('x'):
                try:
                    # Gérer les formats de nombres
                    value = text[:-1].replace(',', '.').strip()
                    score = float(value)
                    scores.append(score)
                except ValueError:
                    logging.warning(f"Valeur non numérique: {text}")
                    continue
        return scores
    
    except Exception as e:
        logging.error(f"Erreur historique: {str(e)}")
        return []

def run_scraper():
    with sync_playwright() as p:
        # Optimisation des ressources
        browser = p.firefox.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--no-sandbox",
                "--disable-images",
                "--disable-fonts",
                "--mute-audio",  # Correction désactivation son
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 800, "height": 600},
            ignore_https_errors=True,
            java_script_enabled=True
        )
        
        page = context.new_page()
        
        try:
            # URL corrigée
            page.goto(
                "https://1win.com.ci/casino/play/v_1winGames:LuckyJet", 
                timeout=60000,
                wait_until="domcontentloaded"
            )
            logging.info("Page principale chargée")

            # Sélecteur d'iframe robuste
            iframe_selectors = [
                'iframe.CasinoGameFrame_root_V6yFR',
                'iframe[src*="luckyjet"]',
                'iframe[src*="lucky"]',
                'div.CasinoOneWinGame_game_goAwv iframe'
            ]
            
            iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=30000)
            frame = iframe.content_frame()
            logging.info("Iframe localisée")

            # Désactivation des éléments non essentiels
            page.evaluate("""
                document.querySelectorAll('video, audio').forEach(e => e.remove());
                document.body.style.animation = 'none';
                document.body.style.transition = 'none';
            """)

            # Boucle principale avec logique métier intacte
            start_time = time.time()
            while True:
                try:
                    # Vérification que l'iframe est toujours valide
                    if frame.is_detached():
                        logging.warning("Iframe détachée, rechargement de la page")
                        page.reload()
                        iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=30000)
                        frame = iframe.content_frame()
                    
                    historical_data = get_history_data(frame)
                    
                    if not historical_data:
                        logging.warning("Aucune donnée historique. Réessai dans 10s")
                        time.sleep(10)
                        continue
                    
                    # Sauvegarde des données brutes (10 premiers éléments)
                    save_data(historical_data[:10])
                    print(f"Nouvelles données: {historical_data[:10]}")
                    
                    # Détermination du nouveau score avec logique originale
                    ScoreDate = date.today()
                    ScoreHeure = datetime.now().time()
                    
                    NewScore = (
                        float(historical_data[0]) 
                        if float(historical_data[0]) > 0 
                        else float(historical_data[1])
                        )
                    
                    # Récupération des 10 derniers scores
                    TenLastScore = getTenLastScoreInArray()
                    
                    # Gestion des listes vides
                    if not TenLastScore:
                        TenLastScore = [1.0]
                        logging.warning("DB vide - Valeur par défaut utilisée")
                    
                    # Calculs statistiques
                    MoyenneMobile = statistics.mean(TenLastScore)
                    EcartType = statistics.stdev(TenLastScore) if len(TenLastScore) > 1 else 0
                    
                    # Classification du score
                    if NewScore < 2:
                        ScoreType = 'Faible'
                    elif 2 <= NewScore < 4.59:
                        ScoreType = 'Moyenne'
                    elif 5 <= NewScore < 9.9:
                        ScoreType = 'Bonne'
                    elif 10 <= NewScore < 49.9:
                        ScoreType = 'Bonne-49'
                    elif 50 <= NewScore < 99.9:
                        ScoreType = 'Bonne-99'
                    elif NewScore >= 100:
                        ScoreType = 'Jackpot'
                    
                    # Vérification et enregistrement
                    last_score_db = getLastScore()
                    
                    if NewScore != last_score_db:
                        if addLastScore([
                            str(ScoreDate),
                            str(ScoreHeure),
                            NewScore,
                            ScoreType,
                            MoyenneMobile,
                            EcartType
                        ]) == 0:
                            print(f"\nNouveau score: {NewScore}x\nHeure: {ScoreHeure}\n")
                        else:
                            print("Erreur d'enregistrement")
                    else:
                        print(f"\nScore inchangé: {last_score_db}x\n")
                        print(f"10 derniers scores: {TenLastScore}\n")
                    
                    # Attente optimisée
                    wait_time = random.uniform(3, 5)
                    print(f"Attente de {wait_time:.2f}s avant vérification...")
                    time.sleep(wait_time)
                    
                    # Redémarrage périodique
                    if time.time() - start_time > 43200:  # 12 heures
                        logging.info("Redémarrage périodique du navigateur")
                        break
                        
                except Exception as e:
                    logging.error(f"Erreur dans la boucle: {str(e)}")
                    time.sleep(10)
        
        except Exception as e:
            logging.critical(f"Erreur critique: {str(e)}")
        finally:
            browser.close()

# Point d'entrée avec redémarrage automatique
if __name__ == "__main__":
    while True:
        try:
            print('Démarrage du scraper...')
            run_scraper()
        except Exception as e:
            logging.error(f"Crash du script: {str(e)}. Redémarrage dans 60s.")
            time.sleep(60)