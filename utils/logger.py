import logging
import os
from datetime import datetime

# Loqların saxlanılacağı qovluğu yoxlayırıq
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Günlük fayl adı (Məsələn: waygo_2026-08-08.log)
log_filename = datetime.now().strftime("waygo_%Y-%m-%d.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

def get_logger(name: str):
    """
    Hər modul üçün xüsusi loglayıcı qaytarır.
    Həm terminala (console), həm də .log faylına yazır.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Əgər əvvəldən handler varsa, ikiqat yazmaması üçün təmizləyirik
    if not logger.handlers:
        # Fayla yazan handler
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Terminala çıxaran handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format: Tarix - Modul Adı - Səviyyə - Mesaj
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
