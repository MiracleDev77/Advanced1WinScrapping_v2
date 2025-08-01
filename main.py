from playwright.sync_api import sync_playwright
import time,random,json,logging,statistics
from datetime import datetime,date
from package import *


# Configuration du logging
logging.basicConfig(filename="scraping.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def save_data(datas, filename="data.json"):
    with open(filename, "a") as f:
        f.write(json.dumps(datas) + "\n")

def run_script():
    with sync_playwright() as p:
        # Lancer Firefox en mode headless avec des optimisations
        browser = p.firefox.launch(headless=True, args=[
            "--disable-images",  # Désactiver les images
            "--disable-fonts",   # Désactiver les polices
            "--disable-audio",  # Désactiver la lecture audio
        ])
        context = browser.new_context(
            viewport={"width": 800, "height": 600},  # Réduire la taille du viewport
            ignore_https_errors=True,
        )
        page = context.new_page()
        iframe_selectors = [
            'iframe[src*="lucky/onewin/"]',  # Le plus spécifique
            'iframe[src*="luckyjet"]',
            'iframe[src*="lucky"]',
            'iframe.CasinoGameFrame_root_V6yFR',
            'iframe.CasinoGame_game_JenRc',
            'div.CasinoOneWinGame_game_goAwv iframe'  # Ancien sélecteur
        ]
        
        try:
            # Charger la page
            page.goto("https://1win.com.ci/casino/play/1play_1play_luckyjet")
            
            print('Detection du frame...')
            # Attendre que l'iframe soit chargé
            #iframe = page.wait_for_selector('div.CasinoOneWinGame_game_goAwv iframe', timeout=60000) #1mn
            #iframe = page.wait_for_selector('iframe.CasinoGameFrame_root_V6yFR, iframe.CasinoGame_game_JenRc', timeout=60000) #1mn
            iframe = page.wait_for_selector(','.join(iframe_selectors), timeout=60000)

            frame = iframe.content_frame()
            
            # Désactiver les animations, les vidéos et la lecture audio
            page.evaluate("""
                document.body.style.animation = 'none';
                document.body.style.transition = 'none';
                for (const video of document.querySelectorAll('video')) {
                    video.pause();
                    video.remove();
                }
                for (const audio of document.querySelectorAll('audio')) {
                    audio.pause();
                    audio.remove();
                }
            """)
            
            # Boucle principale pour récupérer les données
            print("Recherche des données historiques...")
            start_time = time.time()
            while True:
                print('Checking...')
                datas = []
                for i in range(10): #0-9 pour plus de rapidité  # Les IDs vont de 0 à 29
                    try:
                        item = frame.wait_for_selector(f'#history-item-{i}', timeout=5000) #5 secondes (Vu que le navigateur est lancé)
                        value = item.inner_text()
                        datas.append(float(value[:-1].strip()))  # Supprimer le "x"
                    except Exception as e:
                        logging.warning(f"Élément history-item-{i} non trouvé : {e}")
                        continue

                # logging.info(f"Données récupérées : {datas}")
                save_data(datas)
                print(f"Nouvelles données: {datas}")
                ScoreDate = date.today()
                ScoreHeure = datetime.now().time()

                NewScore = float(datas[0]) if float(datas[0]) > 0 else float(datas[1])

                TenLastScore = getTenLastScoreInArray()

                # Si la liste est vide, utilisez une liste par défaut (par exemple [1.0])
                if not TenLastScore:
                    TenLastScore = [1.0]  # Valeur par défaut pour éviter l'erreur
                    logging.warning("La base de données est vide, utilisation d'une valeur par défaut")

                MoyenneMobile = statistics.mean(TenLastScore)
                EcartType = statistics.stdev(TenLastScore) if len(TenLastScore) > 1 else 0

                if NewScore <2:
                    ScoreType = 'Faible';
                elif  2 <= NewScore < 4.59:
                    ScoreType = 'Moyenne';
                elif 5 <= NewScore < 9.9:
                    ScoreType = 'Bonne';
                elif 10 <=NewScore< 49.9 :
                    ScoreType = 'Bonne-49';
                elif 50 <= NewScore< 99.9 :
                    ScoreType = 'Bonne-99';
                elif NewScore > 100:
                    ScoreType = 'Jackpot';


                last_score = getLastScore()

                if NewScore != last_score:#Si le nouveau score n'est pas le dernier score enregistré | cela arrive quand l'avion continu de s'envoler
                    if addLastScore([str(ScoreDate),str(ScoreHeure),NewScore,ScoreType,MoyenneMobile,EcartType]) == 0:
                        print(f"\nNew Score: {NewScore}\nHeure: {ScoreHeure}\n")
                    else:
                        print("erreur signalé")

                elif NewScore == getLastScore():
                    print(f"\nL'avion est toujours dans les aires\nDernier Score:{getLastScore()}\n")
                    print(f"Les 10 derniers scores: {TenLastScore}\n\n")
                
                # Attendre un temps aléatoire entre 1 et 6 secondes
                randomTime = 5#random.randint(3, 5)
                print(f"Attente de {randomTime} seconde pour checker...") 
                time.sleep(randomTime)
                
                # Redémarrer le navigateur toutes les 24 heures
                if time.time() - start_time > 86400:  # 24 heures en secondes
                    browser.close()
                    browser = p.firefox.launch(headless=True)
                    start_time = time.time()
        
        except Exception as e:
            logging.error(f"Une erreur critique s'est produite : {e}")
            print(f"Une erreur critique s'est produite : {e}")
        finally:
            browser.close()

# Redémarrer le script en cas d'erreur critique
while True:
    try:
        print('Starting...')
        run_script()
    except Exception as e:
        logging.error(f"Redémarrage du script après une erreur : {e}")
        time.sleep(60)  # Attendre 1 minute avant de redémarrer