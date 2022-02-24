const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
    title: 'The Computational System Design Language',
    tagline: 'The Language of Multidisciplinary Design Optimization',
    url: 'https://lsdolab.github.io',
    baseUrl: '/csdl/',
    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'warn',
    favicon: 'img/favicon.svg',
    organizationName: 'lsdolab',
    projectName: 'csdl',
    trailingSlash: 'false',
    presets: [
        [
            '@docusaurus/preset-classic',
            /** @type {import('@docusaurus/preset-classic').Options} */
            ({
                docs: {
                    sidebarPath: require.resolve('./sidebars.js'),
                    // Please change this to your repo.
                    editUrl: 'https://github.com/lsdolab/csdl/edit/main/website/',
                    remarkPlugins: [math],
                    rehypePlugins: [katex],
                },
                blog: {
                    showReadingTime: true,
                    // Please change this to your repo.
                    editUrl: 'https://github.com/lsdolab/csdl/edit/main/website/blog/',
                },
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
            }),
        ],
    ],
    stylesheets: [
        {
            href: "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css",
            integrity: "sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc",
            crossorigin: "anonymous",
        },
    ],
    themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            hideableSidebar: true,
            // announcementBar: {
            //     id: 'announcementBar-2', // Increment on change
            // TODO: make an announcement eventually
            //     content: `If you like CSDL, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/lsdolab/csdl">GitHub</a> and engage with us on <a target="_blank" rel="noopener noreferrer" href="https://csdl.zulipchat.com/" >Zulip</a>`,
            // },
            navbar: {
                logo: {
                    alt: 'CSDL',
                    src: 'img/logo.svg',
                    srcDark: 'img/logo-dark.svg',
                },
                items: [
                    {
                        to: '/faq',
                        label: 'FAQ',
                        position: 'right'
                    },
                    {
                        type: 'doc',
                        docId: 'intro',
                        position: 'right',
                        label: 'Docs',
                    },
                    {
                        to: '/news',
                        label: 'News',
                        position: 'right'
                    },
                    {
                        to: '/community',
                        label: 'Community',
                        position: 'right'
                    },
                    {
                        to: '/publications',
                        label: 'Publications',
                        position: 'right'
                    },
                    {
                        to: '/blog',
                        label: 'Blog',
                        position: 'right'
                    },
                ],
            },
            footer: {
                style: 'dark',
                links: [
                    {
                        title: 'Docs',
                        items: [
                            {
                                label: 'Tutorial',
                                to: '/docs/tutorial/install',
                            },
                            {
                                label: 'CSDL by Example',
                                to: '/docs/examples/intro',
                            },
                            {
                                label: 'Language Reference',
                                to: '/docs/lang_ref/model',
                            },
                            {
                                label: 'Developer API',
                                to: '/docs/developer/api',
                            },
                        ],
                    },
                    {
                        title: 'Community',
                        items: [{
                            label: 'Stack Overflow',
                            href: 'https://stackoverflow.com/questions/tagged/csdl-mdo',
                        },
                        {
                            label: 'Zulip',
                            href: 'https://csdl.zulipchat.com/',
                        },
                        ],
                    },
                    {
                        title: 'More',
                        items: [
                            {
                                label: 'Publications',
                                to: '/publications',
                            },
                            {
                                label: 'Blog',
                                to: '/blog',
                            },
                            {
                                label: 'GitHub',
                                href: 'https://github.com/lsdolab/csdl',
                            },
                            {
                                label: 'LSDO Lab',
                                href: 'https://lsdo.eng.ucsd.edu/',
                            },
                        ],
                    },
                    {
                        title: 'Legal',
                        items: [
                            {
                                label: 'LGPLv3 License',
                                href: 'https://github.com/LSDOlab/csdl/blob/master/LICENSE.md',
                            },
                        ],
                    },
                ],
            },
            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
            },
        }),
});
