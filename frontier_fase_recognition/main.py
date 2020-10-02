from person_processing import Person_processing


def main():
    streamer = Person_processing()

    try:
        streamer.process()
    except:
        streamer.log.error("Error: Video_processing starting fail")
        streamer.error_log.error("Error: Video_processing starting fail", exc_info=True)
        print("Error: Video_processing starting fail")


main()
