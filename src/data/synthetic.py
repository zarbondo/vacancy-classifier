# синтетические вакансии в схеме hh.ru: нужны только для --smoke,
# чтобы прогнать весь пайплайн без похода в сеть
import numpy as np
import polars as pl

from src.data.prepare import GRADES

ROLES = {
    "Программист, разработчик": ["python", "go", "сервис", "api", "микросервисы", "postgresql", "docker", "код"],
    "Дата-сайентист": ["ml", "модель", "pytorch", "эксперименты", "метрики", "градиентный бустинг", "признаки"],
    "Аналитик данных": ["sql", "дашборд", "отчётность", "метрики", "ab-тесты", "выгрузки", "tableau"],
    "DevOps-инженер": ["kubernetes", "ci cd", "мониторинг", "terraform", "инфраструктура", "ansible"],
    "Тестировщик": ["тест-кейсы", "автотесты", "регресс", "баги", "qa", "selenium"],
}
GRADE_WORDS = {
    "noExperience": ["стажёр", "junior", "без опыта", "обучим", "начинающий"],
    "between1And3": ["middle", "опыт от года", "самостоятельно", "уверенное знание"],
    "between3And6": ["senior", "опыт от 3 лет", "проектирование", "менторство"],
    "moreThan6": ["lead", "опыт от 6 лет", "архитектура", "руководство командой", "принятие решений"],
}


def generate(n, seed):
    rng = np.random.default_rng(seed)
    roles = list(ROLES)
    rows = []
    for i in range(n):
        role = roles[rng.integers(len(roles))]
        grade = GRADES[rng.integers(len(GRADES))]
        words = list(rng.choice(ROLES[role], size=5, replace=True))
        # в 20% вакансий слова грейда берём от соседнего: на hh описание часто
        # не совпадает с проставленным работодателем опытом, и задача не бывает идеальной
        gi = GRADES.index(grade)
        if rng.random() < 0.2:
            gi = min(max(gi + int(rng.choice([-1, 1])), 0), len(GRADES) - 1)
        words += list(rng.choice(GRADE_WORDS[GRADES[gi]], size=2, replace=True))
        rng.shuffle(words)
        rows.append({"id": i, "text": f"{role}. Требования: " + ", ".join(words),
                     "experience": grade, "role_id": str(roles.index(role)), "role_name": role,
                     "area": "Москва", "employer": f"Компания {i % 50}"})
    print("синтетика: %d вакансий" % n)
    return pl.DataFrame(rows)
