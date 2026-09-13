import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import soundfile as sf


def loadAudio(inputFile, windowSize, timeStart, timeEnd):
	data, sampleRate = sf.read(inputFile)
	if timeEnd >= 0:
		data = data[int(timeStart * sampleRate):int(timeEnd * sampleRate)]
	if data.ndim == 2:
		data = np.mean(data, axis=1)

	A_max = np.max(np.abs(data))
	data = data / A_max

	samplesPerSnippet = int(np.floor(sampleRate * windowSize))
	return data, sampleRate, A_max, samplesPerSnippet


def reconstructSnippet(rows, samples, sampleRate, writePrecision):
	coeffs = np.zeros(samples // 2 + 1, dtype=complex)
	for j in range(len(rows)):
		f, m, p = rows[j]
		m = round(m / samples, writePrecision) * samples
		p = round(p, writePrecision)
		coeffs[int(round(f * samples / sampleRate))] += m * np.exp(1j * p)
	return scipy.fft.irfft(coeffs, n=samples)


def reconstructAudioPlain(rows, samplesPerSnippet, sampleRate, writePrecision):
	reconstructed = np.zeros(len(rows) * samplesPerSnippet)
	for i, selection in enumerate(rows):
		reconstructed[i * samplesPerSnippet:(i + 1) * samplesPerSnippet] = reconstructSnippet(
			selection, samplesPerSnippet, sampleRate, writePrecision)
	return reconstructed


def reconstructAudioHann(rows, samplesPerSnippet, sampleRate, writePrecision):
	hop = samplesPerSnippet // 2
	reconstructed = np.zeros((len(rows) - 1) * hop + samplesPerSnippet)
	for i, selection in enumerate(rows):
		reconstructed[i * hop:i * hop + samplesPerSnippet] += reconstructSnippet(
			selection, samplesPerSnippet, sampleRate, writePrecision)
	return reconstructed


def reconstructAudio(rows, samplesPerSnippet, sampleRate, writePrecision, hannVersion):
	if hannVersion:
		return reconstructAudioHann(rows, samplesPerSnippet, sampleRate, writePrecision)
	else:
		return reconstructAudioPlain(rows, samplesPerSnippet, sampleRate, writePrecision)


def cleanPath(sampleDir):
	if os.path.exists(sampleDir):
		for f in os.listdir(sampleDir):
			os.remove(f"{sampleDir}/{f}")
		os.rmdir(sampleDir)
	os.mkdir(sampleDir)


def writeLine(file, rows, samplesPerSnippet, writePrecision):
	for j in range(len(rows)):
		f, m, p = rows[j]
		file.write(f"{j}\t{f:5.0f}\t{m / samplesPerSnippet:.{writePrecision}f}\t{p:.{writePrecision}f}\n")


def writeLineTransposed(file, snippets, samplesPerSnippet, fileIndex, writePrecision):
	js = [j for snippet in snippets for j in range(len(snippet))]
	fs = [row[0] for snippet in snippets for row in snippet]
	ms = [row[1] / samplesPerSnippet for snippet in snippets for row in snippet]
	ps = [row[2] for snippet in snippets for row in snippet]

	file.write(f"k_{fileIndex}=[" + ",".join(f"{v}" for v in js) + "]\n")
	file.write(f"f_{fileIndex}=[" + ",".join(f"{v}" for v in fs) + "]\n")
	file.write(f"a_{fileIndex}=[" + ",".join(f"{v:.{writePrecision}f}" for v in ms) + "]\n")
	file.write(f"p_{fileIndex}=[" + ",".join(f"{v:.{writePrecision}f}" for v in ps) + "]\n")


def writeToFile(sampleDir, rows, snippetsPerFile, samplesPerSnippet, outputAsList, writePrecision):
	cleanPath(sampleDir)
	if outputAsList:
		with open(f"{sampleDir}/samplesList.txt", 'a') as file:
			for i in range(0, len(rows), snippetsPerFile):
				group = rows[i:i + snippetsPerFile]
				writeLineTransposed(file, group, samplesPerSnippet, i // snippetsPerFile, writePrecision)
	else:
		for i, selection in enumerate(rows):
			with open(f"{sampleDir}/sample{i // snippetsPerFile}.txt", 'a') as file:
				writeLine(file, selection, samplesPerSnippet, writePrecision)
				print("Written Segment:", i, "to File:", i // snippetsPerFile)


def plotAudio(original, reconstructed, sampleRate):
	n = min(len(original), len(reconstructed))
	orig = np.asarray(original[:n])
	recon = np.asarray(reconstructed[:n])

	error = orig - recon
	pointsCount = 5000
	bin_size = max(1, n // pointsCount)
	n_bins = n // bin_size

	origReshaped = orig[:n_bins * bin_size].reshape(n_bins, bin_size)
	errReshaped = error[:n_bins * bin_size].reshape(n_bins, bin_size)

	t = (np.arange(n_bins) * bin_size) / sampleRate

	origMin, origMax = origReshaped.min(axis=1), origReshaped.max(axis=1)
	errMin, errMax = errReshaped.min(axis=1), errReshaped.max(axis=1)

	smoothing = 10
	errMin = np.convolve(errMin, np.ones(smoothing) / smoothing, mode="same")
	errMax = np.convolve(errMax, np.ones(smoothing) / smoothing, mode="same")

	fig, ax = plt.subplots(figsize=(12, 5))

	ax.fill_between(t, origMin, origMax, color="tab:blue", alpha=0.5, label="original")
	ax.fill_between(t, errMin, errMax, color="tab:red", alpha=0.5, label="error")

	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Amplitude")
	ax.legend(loc="upper right")

	plt.tight_layout()
	plt.show()


def writeAudio(reconstructedFile, reconstructed, maxAmp, sampleRate):
	sf.write(reconstructedFile, reconstructed * maxAmp, sampleRate)
