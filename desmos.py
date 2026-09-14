import heapq

import numpy as np
import scipy.fft
import scipy.signal

import helpers as helper

# meta settings
HANN_VERSION = True
OUTPUT_AS_LIST = True
INPUT_FILE = "audio/db.wav"
TIME_START = 0  # start time of audio
TIME_END = -1  # end time of audio, leave -1 for no clipping
SAMPLE_DIR = "./samples"  # folder to write freqs to
WRITE_PRECISION = 3  # precision of magnitude and phase in output
RECONSTRUCTED_FILE = "audio/reconstructed.wav"  # file to write reconstructed audio
RECONSTRUCT_AUDIO_FILE = True
PLOT_AUDIO = True

# settings
SAMPLE_BUDGET = 10000  # desmos table/list hard limit, change if desmodder
FreqCut = 4000  # max frequency to retain, increase if echoey and decrease if tinnitus
WINDOW_SIZE = 1  # size of playing snippet in seconds set value in desmos (data/T)
TABLE_COUNT = 2  # number of tables/lists to be generated


def snippetParams(sampleBudget, snippetCount, tableCount):
	snippetsPerFile = int(np.ceil(snippetCount / tableCount))
	samplesPerSnippet = sampleBudget // snippetsPerFile
	return snippetsPerFile, samplesPerSnippet


def fourierBins(snippet, sampleRate, FreqCut, samples):
	coeffs = scipy.fft.rfft(snippet)
	fourier = np.array([scipy.fft.rfftfreq(samples, 1 / sampleRate), np.abs(coeffs), np.angle(coeffs)])
	index = np.searchsorted(fourier[0], FreqCut)
	fourier = fourier[:, :index]
	return np.transpose(fourier)


def selectTopN(fourier, samplesPerSnippet):
	fourier = fourier[fourier[:, 1].argsort()[::-1]]
	fourier = fourier[:samplesPerSnippet, :]
	return fourier[fourier[:, 0].argsort()]


def selectTopWithNeighbors(fourier, samplesPerSnippet):
	top_heap = []
	for j in range(len(fourier)):
		heapq.heappush(top_heap, (-fourier[j][1], j))

	top_set = set()
	while len(top_set) < samplesPerSnippet and len(top_heap) > 0:
		_, j = heapq.heappop(top_heap)
		top_set.add(j)
		if j > 0 and len(top_set) < samplesPerSnippet:
			top_set.add(j - 1)
		if j < len(fourier) - 1 and len(top_set) < samplesPerSnippet:
			top_set.add(j + 1)
	return [fourier[j] for j in sorted(top_set)]


# legacy version
def generatePlain(data, sampleRate, samples, FreqCut, tableCount):
	snippetCount = int(np.ceil(len(data) / samples))
	snippetsPerFile, samplesPerSnippet = snippetParams(SAMPLE_BUDGET, snippetCount, tableCount)
	data = np.pad(data, (0, snippetCount * samples - len(data)), 'constant')

	padding = 0
	rows = []
	for i in range(snippetCount):
		snippet = data[i * samples:(i + 1) * samples]
		selection = selectTopN(fourierBins(snippet, sampleRate, FreqCut, samples), samplesPerSnippet)
		padding += samplesPerSnippet - len(selection)
		selection = np.pad(selection, ((0, samplesPerSnippet - len(selection)), (0, 0)))
		rows.append(selection)
	return rows, snippetsPerFile, padding


# new version
def generateHann(data, sampleRate, samples, FreqCut, tableCount):
	hop = samples // 2
	snippetCount = 2 * int(np.ceil(len(data) / samples)) - 1
	snippetsPerFile, samplesPerSnippet = snippetParams(SAMPLE_BUDGET, snippetCount, tableCount)
	data = np.pad(data, (0, (snippetCount - 1) * hop + samples - len(data)), 'constant')

	hann = scipy.signal.windows.hann(samples, sym=False)
	hann_start = np.concatenate([np.ones(hop), hann[hop:]])
	hann_end = np.concatenate([hann[:hop], np.ones(samples - hop)])

	padding = 0
	rows = []
	for i in range(snippetCount):
		snippet = data[i * hop:i * hop + samples].copy()
		if i == 0:
			snippet *= hann_start
		elif i == snippetCount - 1:
			snippet *= hann_end
		else:
			snippet *= hann

		selection = selectTopWithNeighbors(fourierBins(snippet, sampleRate, FreqCut, samples), samplesPerSnippet)
		padding += samplesPerSnippet - len(selection)
		selection = np.pad(selection, ((0, samplesPerSnippet - len(selection)), (0, 0)))
		rows.append(selection)
	return rows, snippetsPerFile, padding


def generate(data, sampleRate, samples, FreqCut, tableCount, hannVersion):
	if hannVersion:
		return generateHann(data, sampleRate, samples, FreqCut, tableCount)
	else:
		return generatePlain(data, sampleRate, samples, FreqCut, tableCount)


def main():
	data, sampleRate, maxAmp, samplesPerSnippet = helper.loadAudio(INPUT_FILE, WINDOW_SIZE, TIME_START, TIME_END)

	rows, snippetsPerFile, padding = generate(data, sampleRate, samplesPerSnippet, FreqCut, TABLE_COUNT, HANN_VERSION)

	helper.writeToFile(SAMPLE_DIR, rows, snippetsPerFile, samplesPerSnippet, OUTPUT_AS_LIST, WRITE_PRECISION)

	if RECONSTRUCT_AUDIO_FILE or PLOT_AUDIO:
		reconstructed = helper.reconstructAudio(rows, samplesPerSnippet, sampleRate, WRITE_PRECISION, HANN_VERSION)
		print("Reconstructed Audio")

		if PLOT_AUDIO:
			helper.plotAudio(data, reconstructed, sampleRate)
			print("Graph Plotted")

		if RECONSTRUCT_AUDIO_FILE:
			helper.writeAudio(RECONSTRUCTED_FILE, reconstructed, maxAmp, sampleRate)
			print("Audio File Reconstructed")
	print("Padding is :", padding)


if __name__ == "__main__":
	main()
