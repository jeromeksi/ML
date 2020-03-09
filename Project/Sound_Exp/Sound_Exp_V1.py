import pydub 
import numpy as np
from pathlib import Path

# NOT Working :/
path = 'C:\\Users\\jerome.ksigzkiewiez\\Downloads\\sorry_dave.wav'
a = pydub.AudioSegment.from_file(path,format="wav")
y = np.array(a.get_array_of_samples())