from fairseq.tasks.translation import TranslationTask, register_task

from sim_mt.agent_generator import WaitAgentGenerator


@register_task('agent_sim_translation')
class AgentSimTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        super(AgentSimTranslationTask, AgentSimTranslationTask).add_args(parser)
        parser.add_argument('--agent', default='wait-if-worse', choices=['wait-if-worse', 'wait-if-diff', ])
        parser.add_argument('--force-init-read', default=1, type=int)
        parser.add_argument('--rw-output', default=None)
        parser.add_argument('--encode-once', default=False, action='store_true')

    def build_generator(self, args):
        assert not args.left_pad_source
        assert not args.left_pad_target
        if args.agent in ['wait-if-worse', 'wait-if-diff', ]:
            return WaitAgentGenerator(
                args.agent,
                self.target_dictionary,
                args.rw_output,
                encode_once=args.encode_once,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def get_wait_k(self):
        return 100000
