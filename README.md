# Моделирование динамики синаптически связанных пирамидального и корзинчатого нейрона

Содержание файлов проекта.

### BP_dynamics.py:
ОДУ и их интеграторы

### draws.py:
Отрисовка сигналов, фазового портрета, вейвлет-преобразования и автокорреляционной функции

### wavelets.py: 
Основные функции для расчета вейвлет-преобразования Морле и приложений 

### multistability.py:
Функция для расчета $ISI$, гистограммы $ISI$ при разных Н.У., карты режимов

### lyapunov_exponents.py:
Вычисление старшего показателя Ляпунова, его статистики и спектра показателей Ляпунова

### Poincare.py:
Отрисовка отображения Пуанкаре (2D и 3D), бифуркационной диаграммы, а также расчет матрицы монодромии и мультипликаторов как собственные значения матрицы монодромии



## Замечания
Показатели Ляпунова и отображение Пуанкаре можно использовать для любой системы в общем случае. Для этого необходимо определить функцию $calcODE()$, которая будет конструировать ОДУ и интегрировать его.
