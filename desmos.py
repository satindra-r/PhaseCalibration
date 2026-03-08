import numpy as np
import scipy
import scipy as sp
import soundfile as sf


def analyze_and_reconstruct(input, f_cut, st, out):
	data, sample_rate = sf.read(input)
	if data.ndim == 2:
		data = np.mean(data, axis=1)
	A_max = np.max(data)
	samples = int(np.floor(sample_rate * st))
	n_snippets = int(np.ceil(len(data) / samples))
	n_cut = int(np.floor(10000 / n_snippets))
	data = np.pad(data, (0, n_snippets * samples - len(data)), 'constant')
	reconstructed = np.array([])
	file = open("samples.txt", 'w')
	for i in range(n_snippets):
		snippet = data[i * samples:(i + 1) * samples]
		coeffs = sp.fft.rfft(snippet)
		fourier = np.array([sp.fft.rfftfreq(samples, 1 / sample_rate), np.abs(coeffs) / A_max, np.angle(coeffs)])
		index = np.searchsorted(fourier[0], f_cut)
		fourier = fourier[:, :index]
		fourier = np.transpose(fourier)
		fourier = fourier[fourier[:, 1].argsort()[::-1]]
		fourier = fourier[:n_cut, :]
		fourier = fourier[fourier[:, 0].argsort()]
		fourier = np.pad(fourier, ((0, n_cut - len(fourier)), (0, 0)))
		coeffs_reconstructed = np.zeros(samples // 2 + 1, dtype=complex)
		for i in range(len(fourier)):
			f, m, p = fourier[i]
			file.write(f"{i}\t{f:5.0f}\t{m:.8f}\t{p:.8f}\n")
			coeffs_reconstructed[int(round(f * samples / sample_rate))] = A_max * m * np.exp(1j * p)
		reconstructed = np.concatenate([reconstructed, scipy.fft.irfft(coeffs_reconstructed, n=samples)])
	file.close()
	sf.write(out, reconstructed, sample_rate)


if __name__ == "__main__":
	analyze_and_reconstruct(
		"db.wav",
		3000,
		0.5,
		"reconstructed.wav"
	)
