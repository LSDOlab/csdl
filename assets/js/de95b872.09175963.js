"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[397],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return m}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),c=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=c(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),d=c(n),m=a,h=d["".concat(s,".").concat(m)]||d[m]||u[m]||o;return n?r.createElement(h,i(i({ref:t},p),{},{components:n})):r.createElement(h,i({ref:t},p))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var c=2;c<o;c++)i[c]=n[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},6061:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return c},toc:function(){return p},default:function(){return d}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),i=["components"],l={title:"Best Practices"},s=void 0,c={unversionedId:"tutorial/best-practices",id:"tutorial/best-practices",isDocsHomePage:!1,title:"Best Practices",description:"------------------------------------------------------------------------",source:"@site/docs/tutorial/10-best-practices.mdx",sourceDirName:"tutorial",slug:"/tutorial/best-practices",permalink:"/csdl/docs/tutorial/best-practices",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/tutorial/10-best-practices.mdx",tags:[],version:"current",sidebarPosition:10,frontMatter:{title:"Best Practices"},sidebar:"docs",previous:{title:"Advanced",permalink:"/csdl/docs/tutorial/advanced"},next:{title:"Introduction",permalink:"/csdl/docs/examples/intro"}},p=[{value:"Read the error messages",id:"read-the-error-messages",children:[]},{value:"Use the same (compile time) name for the Python Variable object as the (run time) name for the CSDL variable",id:"use-the-same-compile-time-name-for-the-python-variable-object-as-the-run-time-name-for-the-csdl-variable",children:[]},{value:"Prefer Promotion Over Connection",id:"prefer-promotion-over-connection",children:[]},{value:"Don&#39;t redefine variables",id:"dont-redefine-variables",children:[]},{value:"When creating multiple variables in a compile time loop to aggregate later, store them in a list",id:"when-creating-multiple-variables-in-a-compile-time-loop-to-aggregate-later-store-them-in-a-list",children:[]},{value:"Issue connections at the lowest possible level in the model hierarchy",id:"connections",children:[]},{value:"Always assign types to parameters",id:"always-assign-types-to-parameters",children:[]}],u={toc:p};function d(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("hr",null),(0,o.kt)("p",null,"This page provides some best practices to help you avoid making common\nerrors when using CSDL.\nThis section assumes you have read the preceding sections in this\ntutorial.\nThese best practices are not required, but ",(0,o.kt)("em",{parentName:"p"},"highly recommended"),".\nIf you run into any issues because of something you did wrong, the\n",(0,o.kt)("a",{parentName:"p",href:"../troubleshooting"},"Troubleshooting")," will be more appropriate."),(0,o.kt)("h2",{id:"read-the-error-messages"},"Read the error messages"),(0,o.kt)("p",null,(0,o.kt)("inlineCode",{parentName:"p"},"csdl")," has early error messages meant to prevent users from providing\nill-formed or physically meaningless model specifications.\nThe error messages should be helpful when debugging model\nspecifications.\nIdeally, all errors should occur within ",(0,o.kt)("inlineCode",{parentName:"p"},"Model"),", and if ",(0,o.kt)("inlineCode",{parentName:"p"},"Model")," does\nnot emit an error, then the back end will not emit an error and the\ncompilation will be successful."),(0,o.kt)("h2",{id:"use-the-same-compile-time-name-for-the-python-variable-object-as-the-run-time-name-for-the-csdl-variable"},"Use the same (compile time) name for the Python Variable object as the (run time) name for the CSDL variable"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"lift = self.declare_variable('lift')\n")),(0,o.kt)("p",null,"not"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"l = self.declare_variable('lift')\n")),(0,o.kt)("p",null,"This is not required, but it avoids confusion."),(0,o.kt)("h2",{id:"prefer-promotion-over-connection"},"Prefer Promotion Over Connection"),(0,o.kt)("p",null,"CSDL promotes variable names from child models by default.\nIn some cases, promotions are not allowed, and connections are required.\nPromoting more than one variable from more than one child model is\nallowed if both variables are constructed using\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.declare_variable"),".\nIf two variables are outputs, either constructed by\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output"),", or registered using ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.register_output"),", and\nthey have the same name, then they cannot be promoted to the same level\nin the hierarchy."),(0,o.kt)("p",null,"Connections are meant for connecting two variables with different names,\nor at different levels within the model hierarchy."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('child_1.a', 'child_2.b')\nself.connect('child_1.c', 'child_2.grandchild.b')\n")),(0,o.kt)("h2",{id:"dont-redefine-variables"},"Don't redefine variables"),(0,o.kt)("p",null,"Python does not enforce variable immutability, so if a variable is redefined (or\nin Python parlance, if a variable name is bound to a new reference),\nthen the object in memory storing the variable data is not in use in\nlater parts of the code."),(0,o.kt)("p",null,"This means that your ",(0,o.kt)("inlineCode",{parentName:"p"},"Model")," will lose track of the original variable.\nEither the original variable will be ignored during dead code removal,\nor the original variable will not be used in other parts of the code\nwhere it should have been, leading to unexpected/incorrect behavior in\nthe final simulation."),(0,o.kt)("h2",{id:"when-creating-multiple-variables-in-a-compile-time-loop-to-aggregate-later-store-them-in-a-list"},"When creating multiple variables in a compile time loop to aggregate later, store them in a list"),(0,o.kt)("p",null,"In Python, you might be used to mutating"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"for i, obj in enumerate(iterable):\n    v[i] += obj\n")),(0,o.kt)("p",null,"use"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"l = []\nfor i, obj in enumerate(iterable):\n    l.append(obj)\ns = csdl.sum(*l)\n")),(0,o.kt)("p",null,"or better yet,"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"x = self.create_ouput('x', shape=(n,))\n# concatenate some variables in x...\ns = csdl.sum(*filter(lambda x: True, [x[i] for i in range(n)]))\n")),(0,o.kt)("h2",{id:"connections"},"Issue connections at the lowest possible level in the model hierarchy"),(0,o.kt)("p",null,"Instead of"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('A.B.C.x', 'A.B.y')\n")),(0,o.kt)("p",null,"use"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('B.C.x', 'B.y')\n")),(0,o.kt)("p",null,"within the ",(0,o.kt)("inlineCode",{parentName:"p"},"Model")," named ",(0,o.kt)("inlineCode",{parentName:"p"},"'A'"),"."),(0,o.kt)("h2",{id:"always-assign-types-to-parameters"},"Always assign types to parameters"),(0,o.kt)("p",null,"Instead of"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"self.parameters.declare('cubesats')\nself.parameters.declare('groundstations')\nself.parameters.declare('timesteps')\nself.parameters.declare('stepsize')\n")),(0,o.kt)("p",null,"use"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"self.parameters.declare('cubesats', types=dict)\nself.parameters.declare('groundstations', types=dict)\nself.parameters.declare('timesteps', types=int)\nself.parameters.declare('stepsize', types=float)\n")))}d.isMDXComponent=!0}}]);