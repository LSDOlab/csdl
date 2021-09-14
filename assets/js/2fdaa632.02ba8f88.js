"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[6269,1340],{3905:function(e,n,r){r.d(n,{Zo:function(){return g},kt:function(){return u}});var a=r(7294);function t(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function o(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,a)}return r}function i(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?o(Object(r),!0).forEach((function(n){t(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function l(e,n){if(null==e)return{};var r,a,t=function(e,n){if(null==e)return{};var r,a,t={},o=Object.keys(e);for(a=0;a<o.length;a++)r=o[a],n.indexOf(r)>=0||(t[r]=e[r]);return t}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)r=o[a],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(t[r]=e[r])}return t}var s=a.createContext({}),m=function(e){var n=a.useContext(s),r=n;return e&&(r="function"==typeof e?e(n):i(i({},n),e)),r},g=function(e){var n=m(e.components);return a.createElement(s.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},p=a.forwardRef((function(e,n){var r=e.components,t=e.mdxType,o=e.originalType,s=e.parentName,g=l(e,["components","mdxType","originalType","parentName"]),p=m(r),u=t,d=p["".concat(s,".").concat(u)]||p[u]||c[u]||o;return r?a.createElement(d,i(i({ref:n},g),{},{components:r})):a.createElement(d,i({ref:n},g))}));function u(e,n){var r=arguments,t=n&&n.mdxType;if("string"==typeof e||t){var o=r.length,i=new Array(o);i[0]=p;var l={};for(var s in n)hasOwnProperty.call(n,s)&&(l[s]=n[s]);l.originalType=e,l.mdxType="string"==typeof e?e:t,i[1]=l;for(var m=2;m<o;m++)i[m]=r[m];return a.createElement.apply(null,i)}return a.createElement.apply(null,r)}p.displayName="MDXCreateElement"},4359:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return s},contentTitle:function(){return m},metadata:function(){return g},toc:function(){return c},default:function(){return u}});var a=r(7462),t=r(3366),o=(r(7294),r(3905)),i=r(1619),l=["components"],s={},m="Average of a Single Matrix along Columns",g={unversionedId:"examples/Standard Library/average/ex_average_single_matrix_along0",id:"examples/Standard Library/average/ex_average_single_matrix_along0",isDocsHomePage:!1,title:"Average of a Single Matrix along Columns",description:"This is an example of computing the average of a single matrix input along the",source:"@site/docs/examples/Standard Library/average/ex_average_single_matrix_along0.mdx",sourceDirName:"examples/Standard Library/average",slug:"/examples/Standard Library/average/ex_average_single_matrix_along0",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_single_matrix_along0",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/average/ex_average_single_matrix_along0.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Average of a Single Matrix",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_single_matrix"},next:{title:"Average of a Single Matrix along Rows",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_single_matrix_along1"}},c=[],p={toc:c};function u(e){var n=e.components,r=(0,t.Z)(e,l);return(0,o.kt)("wrapper",(0,a.Z)({},p,r,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"average-of-a-single-matrix-along-columns"},"Average of a Single Matrix along Columns"),(0,o.kt)("p",null,"This is an example of computing the average of a single matrix input along the\ncolumns of the matrix."),(0,o.kt)(i.default,{mdxType:"WorkedExample"}))}u.isMDXComponent=!0},1619:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return m},toc:function(){return g},default:function(){return p}});var a=r(7462),t=r(3366),o=(r(7294),r(3905)),i=["components"],l={},s=void 0,m={unversionedId:"worked_examples/ex_average_single_matrix_along0",id:"worked_examples/ex_average_single_matrix_along0",isDocsHomePage:!1,title:"ex_average_single_matrix_along0",description:"`py",source:"@site/docs/worked_examples/ex_average_single_matrix_along0.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_average_single_matrix_along0",permalink:"/csdl/docs/worked_examples/ex_average_single_matrix_along0",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_average_single_matrix_along0.mdx",tags:[],version:"current",frontMatter:{}},g=[],c={toc:g};function p(e){var n=e.components,r=(0,t.Z)(e,i);return(0,o.kt)("wrapper",(0,a.Z)({},c,r,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleSingleMatrixAlong0(Model):\n    def define(self):\n        n = 3\n        m = 6\n\n        # Declare a matrix of shape 3x6 as input\n        M1 = self.declare_variable(\n            'M1',\n            val=np.arange(n * m).reshape((n, m)),\n        )\n\n        # Output the axiswise average of matrix M1 along the columns\n        self.register_output(\n            'single_matrix_average_along_0',\n            csdl.average(M1, axes=(0, )),\n        )\n\n\nsim = Simulator(ExampleSingleMatrixAlong0())\nsim.run()\n\nprint('M1', sim['M1'].shape)\nprint(sim['M1'])\nprint('single_matrix_average_along_0',\n      sim['single_matrix_average_along_0'].shape)\nprint(sim['single_matrix_average_along_0'])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-M1",metastring:"(3, 6)","(3,":!0,"6)":!0},"[[ 0.  1.  2.  3.  4.  5.]\n [ 6.  7.  8.  9. 10. 11.]\n [12. 13. 14. 15. 16. 17.]]\nsingle_matrix_average_along_0 (6,)\n[ 6.  7.  8.  9. 10. 11.]\n")))}p.isMDXComponent=!0}}]);