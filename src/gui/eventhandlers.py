def _bindEventHandler(handler, **kwargs):
    """Creates an event handler by taking a function that takes many arguments
    and using kwargs to create a function that only takes in one argument"""
    def f(e):
        return handler(e, **kwargs)
    return f