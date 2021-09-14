"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[8870],{3905:function(e,n,t){t.d(n,{Zo:function(){return c},kt:function(){return m}});var r=t(7294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function s(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function a(e,n){if(null==e)return{};var t,r,i=function(e,n){if(null==e)return{};var t,r,i={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var u=r.createContext({}),l=function(e){var n=r.useContext(u),t=n;return e&&(t="function"==typeof e?e(n):s(s({},n),e)),t},c=function(e){var n=l(e.components);return r.createElement(u.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},_=r.forwardRef((function(e,n){var t=e.components,i=e.mdxType,o=e.originalType,u=e.parentName,c=a(e,["components","mdxType","originalType","parentName"]),_=l(t),m=i,d=_["".concat(u,".").concat(m)]||_[m]||p[m]||o;return t?r.createElement(d,s(s({ref:n},c),{},{components:t})):r.createElement(d,s({ref:n},c))}));function m(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var o=t.length,s=new Array(o);s[0]=_;var a={};for(var u in n)hasOwnProperty.call(n,u)&&(a[u]=n[u]);a.originalType=e,a.mdxType="string"==typeof e?e:i,s[1]=a;for(var l=2;l<o;l++)s[l]=t[l];return r.createElement.apply(null,s)}return r.createElement.apply(null,t)}_.displayName="MDXCreateElement"},7561:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return a},contentTitle:function(){return u},metadata:function(){return l},toc:function(){return c},default:function(){return _}});var r=t(7462),i=t(3366),o=(t(7294),t(3905)),s=["components"],a={},u="Einstein Summation New API",l={unversionedId:"std_lib_ref/Linear Algebra/einsum_new",id:"std_lib_ref/Linear Algebra/einsum_new",isDocsHomePage:!1,title:"Einstein Summation New API",description:"This is evaluates the einstein summation on the operands.",source:"@site/docs/std_lib_ref/Linear Algebra/einsum_new.mdx",sourceDirName:"std_lib_ref/Linear Algebra",slug:"/std_lib_ref/Linear Algebra/einsum_new",permalink:"/csdl/docs/std_lib_ref/Linear Algebra/einsum_new",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/std_lib_ref/Linear Algebra/einsum_new.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Reshape",permalink:"/csdl/docs/std_lib_ref/Array Operations/reshape"},next:{title:"Einstein Summation Numpy API",permalink:"/csdl/docs/std_lib_ref/Linear Algebra/einsum_old"}},c=[{value:"Inner Products",id:"inner-products",children:[]},{value:"Outer Products",id:"outer-products",children:[]},{value:"Reorder Operations",id:"reorder-operations",children:[]},{value:"Summation Operations",id:"summation-operations",children:[]},{value:"Special Operations",id:"special-operations",children:[]}],p={toc:c};function _(e){var n=e.components,t=(0,i.Z)(e,s);return(0,o.kt)("wrapper",(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"einstein-summation-new-api"},"Einstein Summation New API"),(0,o.kt)("p",null,"This is evaluates the einstein summation on the operands.\nIt is analogous to numpy.einsum, and uses the same notation."),(0,o.kt)("p",null,"Examples for all the possible use cases are provided below."),(0,o.kt)("p",null,".. autofunction:: csdl.std.einsum_new_api.einsum_new_api"),(0,o.kt)("h2",{id:"inner-products"},"Inner Products"),(0,o.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,o.kt)("p",null,"   ex_einsum_new_inner_vector_vector.rst\nex_einsum_new_inner_vector_vector_sparse.rst\nex_einsum_new_inner_tensor_vector.rst\nex_einsum_new_inner_tensor_vector_sparse.rst"),(0,o.kt)("h2",{id:"outer-products"},"Outer Products"),(0,o.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,o.kt)("p",null,"   ex_einsum_new_outer_vector_vector.rst\nex_einsum_new_outer_vector_vector_sparse.rst\nex_einsum_new_outer_tensor_vector.rst\nex_einsum_new_outer_tensor_vector_sparse.rst"),(0,o.kt)("h2",{id:"reorder-operations"},"Reorder Operations"),(0,o.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,o.kt)("p",null,"   ex_einsum_new_reorder_matrix.rst\nex_einsum_new_reorder_matrix_sparse.rst\nex_einsum_new_reorder_tensor.rst\nex_einsum_new_reorder_tensor_sparse.rst"),(0,o.kt)("h2",{id:"summation-operations"},"Summation Operations"),(0,o.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,o.kt)("p",null,"   ex_einsum_new_vector_summation.rst\nex_einsum_new_vector_summation_sparse.rst\nex_einsum_new_tensor_summation.rst\nex_einsum_new_tensor_summation_sparse.rst"),(0,o.kt)("h2",{id:"special-operations"},"Special Operations"),(0,o.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,o.kt)("p",null,"   ex_einsum_new_multiple_vector_summation.rst\nex_einsum_new_multiple_vector_summation_sparse.rst"))}_.isMDXComponent=!0}}]);