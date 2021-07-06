from six import iteritems

from sphinx_auto_embed.directive import Directive


class DirectiveEmbedMethodsTable(Directive):
    """
    Directive for embedding a table from an OptionsDictionary instance.
    """

    NAME = 'embed-methods-table'
    NUM_ARGS = 0

    def run(self, file_dir, file_name, embed_num_indent, args):
        from ozone.methods_list import family_names, method_families, get_method

        lines = []
        lines.append(' ' * embed_num_indent
            + '.. list-table:: List of integration methods\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':header-rows: 1\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':widths: 20, 20, 20, 20, 20\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':stub-columns: 0\n')
        lines.append('\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '*  -  Method name\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Order\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Num. stages\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Step vec. size\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Type\n')

        for family_name in family_names:
            for method_name in method_families[family_name]:
                method = get_method(method_name)

                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '*  -  %s\n' % method_name)
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '   -  %s\n' % method.order)
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '   -  %s\n' % method.num_stages)
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '   -  %s\n' % method.num_values)
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '   -  %s\n' % ('explicit' if method.explicit else 'implicit'))

            lines.append(' ' * embed_num_indent
                + ' ' * 2 + '*  -\n')
            lines.append(' ' * embed_num_indent
                + ' ' * 2 + '   -\n')
            lines.append(' ' * embed_num_indent
                + ' ' * 2 + '   -\n')
            lines.append(' ' * embed_num_indent
                + ' ' * 2 + '   -\n')
            lines.append(' ' * embed_num_indent
                + ' ' * 2 + '   -\n')

        return lines
