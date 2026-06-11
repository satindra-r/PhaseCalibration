import os
import numpy as np
import scipy
import scipy as sp
import soundfile as sf


def analyze_and_reconstruct(input, f_cut, st, times, timee, count, out):
	data, sample_rate = sf.read(input)
	if timee >= 0:
		data = data[int(times * sample_rate):int(timee * sample_rate)]
	if data.ndim == 2:
		data = np.mean(data, axis=1)
	A_max = np.max(np.abs(data))
	data = data / A_max
	samples = int(np.floor(sample_rate * st))
	n_snippets = int(np.ceil(len(data) / samples))
	rollover = int(np.ceil(n_snippets / count))
	n_cut = 10000 // rollover
	print(n_cut, n_snippets, rollover)
	padding = 0
	data = np.pad(data, (0, n_snippets * samples - len(data)), 'constant')
	reconstructed = np.array([])
	if os.path.exists("./samples"):
		for f in os.listdir("./samples"):
			os.remove(f"./samples/{f}")
			print(f)
		os.rmdir("./samples")
	os.mkdir("./samples")
	for i in range(n_snippets):
		file_count = (i // rollover)
		file = open(f"./samples/sample{file_count}.txt", 'a')
		snippet = data[i * samples:(i + 1) * samples]
		coeffs = sp.fft.rfft(snippet)
		fourier = np.array([sp.fft.rfftfreq(samples, 1 / sample_rate), np.abs(coeffs), np.angle(coeffs)])
		index = np.searchsorted(fourier[0], f_cut)
		fourier = fourier[:, :index]
		fourier = np.transpose(fourier)
		fourier = fourier[fourier[:, 1].argsort()[::-1]]
		fourier = fourier[:n_cut, :]
		fourier = fourier[fourier[:, 0].argsort()]
		padding += n_cut - len(fourier)
		fourier = np.pad(fourier, ((0, n_cut - len(fourier)), (0, 0)))
		coeffs_reconstructed = np.zeros(samples // 2 + 1, dtype=complex)
		for j in range(len(fourier)):
			f, m, p = fourier[j]
			m = m / samples
			file.write(f"{j}\t{f:5.0f}\t{m:.8f}\t{p:.8f}\n")
			coeffs_reconstructed[int(round(f * samples / sample_rate))] = A_max * m * samples * np.exp(1j * p)
		reconstructed = np.concatenate([reconstructed, scipy.fft.irfft(coeffs_reconstructed, n=samples)])
		file.close()
	sf.write(out, reconstructed, sample_rate)
	return padding


if __name__ == "__main__":
	print("padding is :",
	      analyze_and_reconstruct(
			  "rr.mp3",
			  4000,
			  0.5,
			  0,
			  -1,
			  4,
			  "reconstructed.wav"
		  )
	      )
