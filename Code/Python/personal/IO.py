import re
import urllib
import pickle


class InFile(object):
    """
    This puppy is supposed to help w/ reading files that need a function to process...
    """
    def __init__(self, infile, line_func=lambda x: x): # initiate, line_func processes lines in next(self) fcn
        self.line_func = line_func
        self.infile = urllib.request.urlopen(infile)

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def read(self, *args, **kwargs):
        # return next_line if next_line is not None else None
        try:
            yield self.__next__()
        except StopIteration:
            return

    def next(self):
        try:
            line: str = self.infile.readline().decode('utf-8')
            if len(line)==0:
                self.infile.close()
                raise StopIteration
            line = self.line_func(line) # do whatever it was that needed to be done to each line! 
            return line.split()
        except:
            self.infile.close()
            raise StopIteration
            
def pickle_save(obj, filename,*args,**kwargs):
    with open(filename, 'wb+') as filename:
        pickle.dump(obj, filename,*args, **kwargs)
        
def pickle_load(filename, *args, **kwargs):
    with open(filename, 'rb') as filename:
        return pickle.load( filename,*args, **kwargs)   