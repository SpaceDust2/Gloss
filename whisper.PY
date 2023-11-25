# Импортируем библиотеку для работы с аудио
import librosa
import soundfile as sf
import whisper
# Загружаем аудиофайл
audio, sr = librosa.load(r"PathToAudio")

# Определяем порог громкости для тишины
threshold = 0.01

# Находим индекс первого отсчета, который превышает порог
start_index = 0
for i in range(len(audio)):
  if abs(audio[i]) > threshold:
    start_index = i
    break

# Обрезаем аудиофайл с начала до найденного индекса
trimmed_audio = audio[start_index:]

# Сохраняем обрезанный аудиофайл
sf.write("audio_new.mp3", trimmed_audio, sr)

model = whisper.load_model("large-v3")
result = model.transcribe("audio.mp3")
with open("result.txt", "w") as f:
    f.write(result["text"])
print(result["text"])