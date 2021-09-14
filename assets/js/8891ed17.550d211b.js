"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[3692,8427],{3905:function(e,r,n){n.d(r,{Zo:function(){return d},kt:function(){return p}});var t=n(7294);function o(e,r,n){return r in e?Object.defineProperty(e,r,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[r]=n,e}function i(e,r){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),n.push.apply(n,t)}return n}function a(e){for(var r=1;r<arguments.length;r++){var n=null!=arguments[r]?arguments[r]:{};r%2?i(Object(n),!0).forEach((function(r){o(e,r,n[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))}))}return e}function s(e,r){if(null==e)return{};var n,t,o=function(e,r){if(null==e)return{};var n,t,o={},i=Object.keys(e);for(t=0;t<i.length;t++)n=i[t],r.indexOf(n)>=0||(o[n]=e[n]);return o}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)n=i[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var u=t.createContext({}),m=function(e){var r=t.useContext(u),n=r;return e&&(n="function"==typeof e?e(r):a(a({},r),e)),n},d=function(e){var r=m(e.components);return t.createElement(u.Provider,{value:r},e.children)},l={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},c=t.forwardRef((function(e,r){var n=e.components,o=e.mdxType,i=e.originalType,u=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),c=m(n),p=o,_=c["".concat(u,".").concat(p)]||c[p]||l[p]||i;return n?t.createElement(_,a(a({ref:r},d),{},{components:n})):t.createElement(_,a({ref:r},d))}));function p(e,r){var n=arguments,o=r&&r.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=c;var s={};for(var u in r)hasOwnProperty.call(r,u)&&(s[u]=r[u]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var m=2;m<i;m++)a[m]=n[m];return t.createElement.apply(null,a)}return t.createElement.apply(null,n)}c.displayName="MDXCreateElement"},5372:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return u},contentTitle:function(){return m},metadata:function(){return d},toc:function(){return l},default:function(){return p}});var t=n(7462),o=n(3366),i=(n(7294),n(3905)),a=n(2313),s=["components"],u={},m="Reordering a Matrix using Einsum",d={unversionedId:"examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix",id:"examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix",isDocsHomePage:!1,title:"Reordering a Matrix using Einsum",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix.mdx",sourceDirName:"examples/Standard Library/einsum_old",slug:"/examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Vector-Vector Outer Product using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_outer_vector_vector_sparse"},next:{title:"Reordering a Matrix using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_reorder_matrix_sparse"}},l=[],c={toc:l};function p(e){var r=e.components,n=(0,o.Z)(e,s);return(0,i.kt)("wrapper",(0,t.Z)({},c,n,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"reordering-a-matrix-using-einsum"},"Reordering a Matrix using Einsum"),(0,i.kt)("p",null,"This is an example of how to properly use the einsum function\nto reorder a matrix."),(0,i.kt)(a.default,{mdxType:"WorkedExample"}))}p.isMDXComponent=!0},2313:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return s},contentTitle:function(){return u},metadata:function(){return m},toc:function(){return d},default:function(){return c}});var t=n(7462),o=n(3366),i=(n(7294),n(3905)),a=["components"],s={},u=void 0,m={unversionedId:"worked_examples/ex_einsum_old_reorder_matrix",id:"worked_examples/ex_einsum_old_reorder_matrix",isDocsHomePage:!1,title:"ex_einsum_old_reorder_matrix",description:"`py",source:"@site/docs/worked_examples/ex_einsum_old_reorder_matrix.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_old_reorder_matrix",permalink:"/csdl/docs/worked_examples/ex_einsum_old_reorder_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_old_reorder_matrix.mdx",tags:[],version:"current",frontMatter:{}},d=[],l={toc:d};function c(e){var r=e.components,n=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,t.Z)({},l,n,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleReorderMatrix(Model):\n    def define(self):\n        shape2 = (5, 4)\n        b = np.arange(20).reshape(shape2)\n        mat = self.declare_variable('b', val=b)\n\n        # Transpose of a matrix\n        self.register_output('einsum_reorder1',\n                             csdl.einsum(mat, subscripts='ij->ji'))\n\n\nsim = Simulator(ExampleReorderMatrix())\nsim.run()\n\nprint('b', sim['b'].shape)\nprint(sim['b'])\nprint('einsum_reorder1', sim['einsum_reorder1'].shape)\nprint(sim['einsum_reorder1'])\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-b",metastring:"(5, 4)","(5,":!0,"4)":!0},"[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]\n [12. 13. 14. 15.]\n [16. 17. 18. 19.]]\neinsum_reorder1 (4, 5)\n[[ 0.  4.  8. 12. 16.]\n [ 1.  5.  9. 13. 17.]\n [ 2.  6. 10. 14. 18.]\n [ 3.  7. 11. 15. 19.]]\n")))}c.isMDXComponent=!0}}]);