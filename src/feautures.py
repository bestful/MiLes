from src import model

# In[1]

class Feauture:
  def __init__(self, model):
    

class Feautures:
  def __init__(self, feautures):
    for model in feautures:
      self.feaututres += [Feauture(model)]

def generator_brut(lst):
  n = len(lst)
  if n == 1:
    yield lst
  elif n == 0:
    yield []
  else:
    el = lst[0]
    tail = lst[1:]
    for it in generator_brut(tail):
        yield [el] + it
        yield it

class su:
  def __init__(self):
    
def g_member(lst):
  from x in lst:


def generator_dfs(verices):
  for x in vis
    if rule(x, v):
      



_generator_brut = generator_brut([1, 2, 3])



print([x for x in _generator_brut])