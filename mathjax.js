document.write([
  '<script type="text/x-mathjax-config">',
    'MathJax.Hub.Config({',
       'tex2jax: {',
         'inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]],',
         'skipTags: ["script", "noscript", "style", "textarea"],',
       '},',
    '});',
  '</script>',
  '<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>'
].join('\n'));
