import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dados da matriz de confusão
cm = np.array([
    [1307,    0,  515],
    [   6,  180,    4],
    [ 396,    0, 1604]
])

# Nomes das classes
class_names = ['white', 'draw', 'black']

# Criar o gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão - Modelo Chess Com GradientBoostingClassifier')
plt.tight_layout()
plt.show()
