let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Sandboxes/pybullet_swarming
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +99 pybullet_swarming/environment/simulator.py
badd +22 pybullet_swarming/flier/swarm_controller.py
badd +19 experiments/cube_scratch.py
badd +58 experiments/single_circle.py
badd +67 pybullet_swarming/utility/shebangs.py
argglobal
%argdel
$argadd ./
edit experiments/single_circle.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '2resize ' . ((&lines * 2 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 1 + 105) / 211)
exe '3resize ' . ((&lines * 2 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 31 + 105) / 211)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 58 - ((31 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
58
normal! 013|
wincmd w
argglobal
enew
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
argglobal
enew
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
exe '2resize ' . ((&lines * 2 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 1 + 105) / 211)
exe '3resize ' . ((&lines * 2 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 31 + 105) / 211)
if exists(':tcd') == 2 | tcd ~/Sandboxes/pybullet_swarming | endif
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFA
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
