import time

start_time = time.time()
display_time = True
display_categories = [1]


def log(s, category=0):
    if category in display_categories:
        if display_time:
            print '{0:.5}: {1}'.format(time.time() - start_time, s)
        else:
            print s