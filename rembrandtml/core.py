
class MLEntityBase(object):
    def __init__(self):
        self.Base_Directory = os.path.abspath(os.path.join(os.getcwd(), '../../..'))

    '''verbosity levels: s(silence) q(quiet) m(moderate) d(debug) (noisy)'''
    def log(self, msg, verbosity='d'):
        if(self.Config.Verbose == verbosity):
            print(msg)