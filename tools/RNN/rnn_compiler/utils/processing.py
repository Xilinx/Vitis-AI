
def param_not_empty(func):
    def wrapper(*args):
        for arg in args:
#            print('-----------------------------------')
#            print(arg)
            if len(arg) == 0:
                raise ValueError('function params should not be empty')
        return func(*args)
    return wrapper
