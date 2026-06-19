#  IBM HR Analytics - Прогнозирование текучести сотрудников

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-✓-green?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-✓-yellow?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-ANN-red?style=flat-square&logo=keras)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---
## Командный проект школы VeraVla, в данном проекте я занималась анализом и обучением моделей.

##  О проекте

Полный цикл разработки ML-модели на датасете **IBM HR Analytics Employee Attrition & Performance** - один из классических датасетов в области People Analytics.

**Цель:** предсказать, уволится ли сотрудник (бинарная классификация по признаку `Attrition`).

Датасет был искусственно сгенерирован на основе типовых HR-сценариев и впервые появился в середине 2010-х годов, когда тема People Analytics начала активно развиваться.

---

##  Этапы работы

### 1. Разведочный анализ данных (EDA)
- Анализ распределения целевой переменной (`Attrition`)
- Выявление дисбаланса классов (~84% / ~16%)
- Анализ корреляций и распределений числовых и категориальных признаков

### 2. Предобработка и Feature Engineering
- Удалены **константные** признаки 
- Удалены **дублирующиеся** и **мультиколлинеарные** признаки
- Созданы **новые информативные переменные**
- Сформировано несколько версий датасета для сравнения влияния feature engineering

### 3. Обучение и сравнение моделей

| Модель | Описание |
|---|---|
| Logistic Regression | ✅ Выбрана как основная - лучший баланс качества и интерпретируемости |
| Random Forest | Протестирована |
| Decision Tree | Протестирована |
| Gradient Boosting | Протестирована |
| XGBoost | Протестирована |
| CatBoost | Протестирована |

### 4. Эксперименты с нейронными сетями (ANN)
- Построены модели на базе **Keras / TensorFlow**
- Хорошее разделение классов, но выявлены признаки **умеренного переобучения**

---

## Ключевая проблема: дисбаланс классов

На протяжении всего исследования главным вызовом оставался **выраженный дисбаланс классов**. Применение `class_weight='balanced'` частично сгладило проблему, однако модели по-прежнему значительно лучше распознавали отрицательный класс (остался в компании), чем положительный (уволился).

---

##  Результаты

- Лучшая модель среди классических алгоритмов - **Logistic Regression**
- Модели продемонстрировали **хорошую способность к обобщению** и могут использоваться как базовое решение
- ANN показали конкурентные результаты, но требуют доработки с точки зрения регуляризации

---

##  Стек технологий

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `CatBoost` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn`

---

##  Датасет

[IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

> Датасет содержит **35 признаков** и **1 470 наблюдений**, включая демографические данные, рабочие характеристики и уровни удовлетворённости сотрудников.
