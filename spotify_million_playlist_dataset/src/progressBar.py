import time


class ProgressBar:
    def __init__(self, total, prefix='Progress', suffix='Complete', decimals=1, length=100, fill='█', printEnd='',
                 timing=False):
        """
        This ProgressBar can be used to check how fast a algorithm is and to know the current progress of the program.
        :param total:       Required - total iterations (Int)
        :param prefix:      Optional - prefix string (Str)
        :param suffix:      Optional - suffix string (Str)
        :param decimals:    Optional - positive number of decimals in percent complete (Int)
        :param length:      Optional - character length of bar (Int)
        :param fill:        Optional - bar fill character (Str)
        :param printEnd:    Optional - end character (e.g. "\r", "\r\n", "" for Windows) (Str)
        :param timing:      Optional - if true, the progressbar calculates the outstanding runtime from the previous
                            runtime and refreshes it in every step.
                            At the end it prints the total runtime of the program.
        """
        self.total = total
        self.gap = max(total // pow(10, decimals + 2), 1)  # used to calculate only in gaps
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.printEnd = printEnd
        self.timing = timing
        '''
        :param stopPrinting:    The print-method will be called too often to just check the last iteration,
                                so we have to end in a range of iteration but wont print the last line too often
        :param startTime:       Used to calculate the outstanding runtime and the total runtime
        '''
        self.stopPrinting = False
        self.startTime = time.time()

    def printProgressBar(self, iteration):
        """
        Call in a loop to create terminal progress bar. Inspired by:
        https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
        :param iteration: Required - current iteration (Int)
        :return: Nothing, print only
        """

        if iteration % self.gap == 0 and not self.stopPrinting:

            percent = round(iteration / float(self.total) * 100, self.decimals)
            filledLength = int(self.length * iteration // self.total)
            bar = self.fill * filledLength + '-' * (self.length - filledLength - 1)
            if self.timing:
                timeCurrent = time.time()
                stepTime = timeCurrent - self.startTime
                timePred = round((100 - percent) / (percent + 1) * stepTime, 1)
                timePredMin = int(timePred // 60)
                timePredSec = int(timePred % 60)
                timePredStr = f'[{timePredMin}:{timePredSec if timePredSec > 9 else "0" + str(timePredSec)} min:sec]'
                print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {timePredStr}', end=self.printEnd)
            else:
                print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end=self.printEnd)
            # Print last line with linebreak on complete
            if percent > 99.5:  # terminate for a range of iterations cause of the fast method-calls
                if self.timing:
                    processTime = round(timeCurrent - self.startTime, 1)
                    processTimeMin = int(processTime // 60)
                    processTimeSec = int(processTime % 60)
                    processTimeStr = f'[{processTimeMin}:' \
                                     f'{processTimeSec if processTimeSec > 9 else "0" + str(processTimeSec)} min:sec]'
                    print(f'\r{self.prefix} |{bar}| {100}% {self.suffix} {processTimeStr}')
                else:
                    print(f'\r{self.prefix} |{bar}| {100}% {self.suffix}')
                self.stopPrinting = True
