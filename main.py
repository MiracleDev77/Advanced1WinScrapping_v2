from playwright.sync_api import sync_playwright
import time, random, json, logging, statistics, os
from datetime import datetime, date
from package import init_database, addLastScore, getLastScore, getTenLastScoreInArray
import numpy as np 

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
    """Récupère robustement les scores historiques de manière optimisée"""
    try:
        # Attendre que l'historique soit chargé avec timeout réduit
        frame.wait_for_selector('[data-testid^="history-item-"]', timeout=5000)
        
        # Récupération directe des éléments sans boucle coûteuse
        history_items = frame.query_selector_all(
            '[data-testid^="history-item-"]:not([data-testid="history-item-undefined"])'
        )
        
        scores = []
        for i in range(min(15, len(history_items))):  # Seulement les 15 premiers éléments
            text = history_items[i].inner_text().strip()
            if text.endswith('x'):
                try:
                    value = text[:-1].replace(',', '.').strip()
                    score = float(value)
                    scores.append(score)
                except ValueError:
                    continue
        return scores
    
    except Exception as e:
        logging.warning(f"Erreur historique: {str(e)}")
        return []

def run_scraper():
    with sync_playwright() as p:
        # Optimisation des ressources avec cache
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--no-sandbox",
                "--disable-images",
                "--disable-fonts",
                "--mute-audio",
                "--disable-audio",
                "--disk-cache-size=1"
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 800, "height": 600},
            ignore_https_errors=True,
            java_script_enabled=True,
            # Activation du cache pour accélérer les chargements
            storage_state="state.json" if os.path.exists("state.json") else None
        )
        
        page = context.new_page()
        
        try:
            # Initialiser la base de données
            init_database()
            
            # Navigation avec stratégie "eager" pour charger plus vite
            page.goto(
                "https://1win.com.ci/casino/play/v_1winGames:LuckyJet", 
                timeout=30000,
                wait_until="domcontentloaded"
            )
            logging.info("Page principale chargée")

            # Sauvegarde de l'état de session pour les prochaines exécutions
            context.storage_state(path="state.json")

            # Sélecteur d'iframe robuste
            iframe_selectors = [
                'iframe.CasinoGameFrame_root_V6yFR',
                'iframe[src*="luckyjet"]',
                'div.CasinoOneWinGame_game_goAwv iframe'
            ]
            
            iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=15000)
            frame = iframe.content_frame()
            logging.info("Iframe localisée")

            # Désactivation des éléments non essentiels
            page.evaluate("""
                const elements = document.querySelectorAll('video, audio, img');
                elements.forEach(e => e.remove());
            """)

            # Variables d'état pour optimisation
            last_valid_scores = []
            last_processed_score = 0
            ten_last_cache = []
            
            # Boucle principale optimisée
            start_time = time.time()
            while True:
                try:
                    # Vérification rapide de l'iframe
                    if frame.is_detached():
                        logging.warning("Iframe détachée, rechargement rapide")
                        page.reload(timeout=20000, wait_until="domcontentloaded")
                        iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=10000)
                        frame = iframe.content_frame()
                    
                    # Récupération historique avec timeout réduit
                    try:
                        historical_data = get_history_data(frame)
                    except:
                        historical_data = last_valid_scores  # Utiliser le dernier état connu
                    
                    if not historical_data:
                        time.sleep(2)
                        continue
                    
                    # Filtrer les scores valides (positifs)
                    valid_scores = [s for s in historical_data if s > 0]
                    
                    if not valid_scores:
                        time.sleep(2)
                        continue
                    
                    # Sauvegarde des données brutes uniquement si changement
                    if valid_scores != last_valid_scores:
                        #save_data(valid_scores[:5])  # Seulement 5 premiers pour réduire I/O
                        print(f"Scores valides: {valid_scores[:5]}")
                        last_valid_scores = valid_scores
                    
                    # Détermination du nouveau score
                    NewScore = valid_scores[0]
                    
                    # Utilisation du cache pour éviter des requêtes SQL constantes
                    if not ten_last_cache:
                        ten_last_cache = getTenLastScoreInArray() or [1.0]
                    
                    # Vérification et enregistrement
                    if NewScore != last_processed_score:
                        # Mise à jour du cache
                        ten_last_cache.insert(0, NewScore)
                        ten_last_cache = ten_last_cache[:10]  # Garder seulement 10 éléments
                        
                        # Calculs statistiques
                        MoyenneMobile = statistics.mean(ten_last_cache)
                        EcartType = statistics.stdev(ten_last_cache) if len(ten_last_cache) > 1 else 0
                        
                        # Enregistrement
                        if addLastScore([
                            str(date.today()),
                            datetime.now().strftime("%H:%M:%S"),
                            NewScore,
                            MoyenneMobile,
                            EcartType
                        ]) == 0:
                            print(f"\n✅ Nouveau score: {NewScore}x")
                            print(f"   Période: {'Favorable' if MoyenneMobile >= 3 else 'Défavorable'}")
                            last_processed_score = NewScore
                        else:
                            print("❌ Erreur d'enregistrement")
                    
                    # Attente optimisée avec backoff dynamique
                    wait_time = 0.5 + random.random()  # Entre 0.5 et 1.5 secondes
                    time.sleep(wait_time)
                    
                    # Rafraîchissement périodique
                    if time.time() - start_time > 1800:  # 30 minutes
                        logging.info("Rafraîchissement périodique de la page")
                        page.reload(timeout=20000, wait_until="domcontentloaded")
                        iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=10000)
                        frame = iframe.content_frame()
                        start_time = time.time()
                        
                except Exception as e:
                    logging.warning(f"Erreur mineure: {str(e)}")
                    time.sleep(3)
        
        except Exception as e:
            logging.critical(f"Erreur critique: {str(e)}", exc_info=True)
        finally:
            try:
                browser.close()
            except:
                pass


# Point d'entrée optimisé
if __name__ == "__main__":
    logging.info('Démarrage du scraper optimisé')
    while True:
        try:
            print('='*50)
            print(' SanF|an Casino Scrapper Pro'.center(50,'★'))
            print('='*50)
            run_scraper()
        except Exception as e:
            logging.error(f"Crash: {str(e)}. Redémarrage dans 10s.", exc_info=True)
            time.sleep(10)