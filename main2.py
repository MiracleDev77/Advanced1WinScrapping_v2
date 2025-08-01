from playwright.sync_api import sync_playwright
import time, random, json, logging, statistics
from datetime import datetime, date
from package import *  # Conserve les fonctions métiers

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
        # Attendre que l'historique soit chargé
        frame.wait_for_selector('[data-testid^="history-item-"]', timeout=5000)
        
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
                "--disable-audio",
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

            # Boucle principale optimisée
            start_time = time.time()
            last_score = None
            while True:
                try:
                    historical_data = get_history_data(frame)
                    
                    if historical_data:
                        current_score = historical_data[0]
                        
                        # Vérification nouveau score
                        if current_score != last_score:
                            logging.info(f"Nouveau score détecté: {current_score}x")
                            last_score = current_score
                            
                            # Classification
                            if current_score < 2:
                                score_type = 'Faible'
                            elif 2 <= current_score < 5:
                                score_type = 'Moyen'
                            elif 5 <= current_score < 10:
                                score_type = 'Bon'
                            elif 10 <= current_score < 50:
                                score_type = 'Très bon'
                            else:
                                score_type = 'Exceptionnel'
                            
                            # Sauvegarde des données brutes
                            save_data(historical_data)
                            
                            # Récupération des 10 derniers scores
                            ten_last_scores = getTenLastScoreInArray() or [1.0]
                            
                            # Calculs statistiques
                            moyenne = statistics.mean(ten_last_scores)
                            ecart_type = statistics.stdev(ten_last_scores) if len(ten_last_scores) > 1 else 0
                            
                            # Enregistrement en base
                            if addLastScore([
                                str(date.today()),
                                str(datetime.now().time()),
                                current_score,
                                score_type,
                                moyenne,
                                ecart_type
                            ]) == 0:
                                print(f"\nNouveau score enregistré: {current_score}x\n")
                            else:
                                logging.error("Erreur d'enregistrement en base")
                            
                            # Affichage des derniers scores
                            print(f"10 derniers scores: {ten_last_scores[:10]}")
                        else:
                            print(f"Score actuel inchangé: {current_score}x")
                    
                    # Attente optimisée
                    wait_time = random.uniform(3, 5)
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