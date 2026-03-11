¿El RMSE disminuyó?
Sí. 

---


Al observar el historial a lo largo de las 200 épocas, 



el RMSE muestra una tendencia a la baja. Esto confirma que el algoritmo de Descenso de Gradiente Estocástico (SGD) funciona correctamente: en cada iteración, los parámetros se ajustan en la dirección que minimiza el error, permitiendo que el modelo "aprenda" la relación matemática que existe entre los caballos de fuerza, el peso y el consumo de combustible.

Limitaciones de entrenar con solo 15 instancias:

---


Construir un modelo predictivo con una muestra tan pequeña presenta serios problemas para su viabilidad:

---



Alto riesgo de Sobreajuste (Overfitting): Con solo 15 ejemplos, el modelo tiende a "memorizar" esos datos específicos en lugar de generalizar el patrón. Esto provoca que funcione bien con los datos de entrenamiento, pero que su error (RMSE) sea considerablemente mayor al evaluar datos nunca antes vistos (el conjunto de prueba).


---


Falta de representatividad: 15 vehículos no logran representar la inmensa diversidad de configuraciones de motores y pesos que existen. El modelo se queda con una perspectiva sumamente limitada del problema.

---



Sensibilidad a valores atípicos (Outliers): En un conjunto de miles de datos, un auto con estadísticas anómalas se diluye. Sin embargo, en un grupo de solo 15, un único vehículo fuera de lo común modificara drásticamente la línea de regresión, sesgando todas las predicciones futuras.