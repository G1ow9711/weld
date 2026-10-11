"""Evaluate the full-coverage fine-tuned checkpoint with fixed event pairing."""
if __name__=='__main__':
    import sys
    from weld_pipeline.sjg_free_sequence_823 import main
    from weld_pipeline.sjg_event_pairs_824 import decode_event_pairs
    if '--checkpoint-run' not in sys.argv:sys.argv.extend(['--checkpoint-run','run44'])
    main(decoder=decode_event_pairs)
