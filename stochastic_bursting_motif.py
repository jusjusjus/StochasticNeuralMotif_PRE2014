#!/usr/bin/python

"""
We thank Dr. Robert H. Clewley for his help in adding 32-bit support.
"""

from pylab import *
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import struct

inline_available = True

try:
	from scipy.weave import inline

except:
	inline_available = False
	print 'Patience please;  computing new trajectory ...'


def architecture():
	"""
	Thanks to Robert:
	Platform- and version-independent function to determine 32- or 64-bit architecture.
	Used primarily to determine need for "-m32" option to C compilers for external library
	compilation, e.g. by AUTO, Dopri, Radau.

	Returns integer 32 or 64.
	"""
	return struct.calcsize("P") * 8

def extra_arch_arg():
	"""
	Thanks to Robert:
	Adds '-m32' flag to list of extra compiler/linker flags,
	based on whether architecture is detected as 32 bit. Otherwise,
	it returns the empty list.
	"""
	if architecture() == 32:
		return ['-m32']

	else:
		return []

# fixed parameter values


C = 0.5 # nF

g_K2 = 30. # nS
g_Na = 160. # nS
g_L = 8. # nS

E_Na = 45. # mV
E_K = -70. # mV
E_L = -46. # mV

tau_K2 = 0.9 # sec
tau_Na = 0.0405 # sec

V_h = -32.5 # mV
V_Na = -30.5 # mV
V_K = 3. # mV

s_h = -0.5 # mV^-1
s_Na = 0.15 # mV^-1
s_K = 0.083 # mV^-1
s_syn = 1. # mV^-1

theta_syn = -40. # mV
I_0 = -6. # nA


C_inv = 1./C
tau_K2_inv = 1./tau_K2
tau_Na_inv = 1./tau_Na

# parameter values to be varied

# g_syn = 5.*10**-2 # nS
E_syn =  -62.5 # mV

# sigma = 0.1 # pA/sqrt(sec)




default_args = ['g_syn']

function_code = """
	#define	C	0.5
	#define	g_K2	30.
	#define	g_Na	160.
	#define	g_L	8.
	#define	E_Na	45.
	#define	E_K	-70.
	#define	E_L	-46.
	#define	tau_K2	0.9
	#define	tau_Na	0.0405
	#define	V_h	-32.5
	#define	V_Na	-30.5
	#define	V_K	3.
	#define	s_h	-0.5
	#define	s_Na	0.15
	#define	s_K	0.083
	#define	s_syn	1.
	#define	theta_syn	-40.
	#define	I_0	-6.
	#define	E_syn	-62.5

	double activation(const double V, const double V_0, const double s)
	{
		return 1./(1.+exp(-s*(V-V_0)));
	};
	
	void derivs_three(const double* y, double* dxdt, const double g_syn)
	{
		double activation_1=activation(y[0], theta_syn, s_syn), activation_2=activation(y[3], theta_syn, s_syn), activation_3=activation(y[6], theta_syn, s_syn);
	
		double
		coupling = g_syn*(E_syn-y[0])*(activation_2+activation_3);
		dxdt[0] = (-g_Na*pow(activation(y[0], V_Na, s_Na), 3.)*y[1]*(y[0]-E_Na)-g_K2*y[2]*y[2]*(y[0]-E_K)-g_L*(y[0]-E_L)+I_0+coupling)/C;
		dxdt[1] = (activation(y[0], V_h, s_h)-y[1])/tau_Na;
		dxdt[2] = (activation(y[0], V_K, s_K)-y[2])/tau_K2;
		
		coupling = g_syn*(E_syn-y[3])*(activation_1+activation_3);
		dxdt[3] = ( -g_Na*pow(activation(y[3], V_Na, s_Na), 3.)*y[4]*(y[3]-E_Na) - g_K2*y[5]*y[5]*(y[3]-E_K) - g_L*(y[3]-E_L)+I_0 + coupling )/C;
		dxdt[4] = (activation(y[3], V_h, s_h)-y[4])/tau_Na;
		dxdt[5] = (activation(y[3], V_K, s_K)-y[5])/tau_K2;
		
		coupling = g_syn*(E_syn-y[6])*(activation_1+activation_2);
		dxdt[6] = (-g_Na*pow(activation(y[6], V_Na, s_Na), 3.)*y[7]*(y[6]-E_Na)-g_K2*y[8]*y[8]*(y[6]-E_K)-g_L*(y[6]-E_L)+I_0+coupling)/C;
		dxdt[7] = (activation(y[6], V_h, s_h)-y[7])/tau_Na;
		dxdt[8] = (activation(y[6], V_K, s_K)-y[8])/tau_K2; 
	}
"""


c_integrate_em = """
	py::tuple output(3*N_integrate);

	unsigned i, j, k;
	double sigma, right_hand_side[9];

	for(j=0; j<3; j++)
		output[j] = y[3*j]; 	

	for(i=1; i<N_integrate; i++)
	{
		for(j=0; j<stride; j++)
		{
			derivs_three(y, right_hand_side, g_syn);
			for(k=0; k<9; k++) y[k] = y[k]+right_hand_side[k]*dt; 			
			y[0] += noise[3*(i*stride+j)];
			y[3] += noise[3*(i*stride+j)+1];
			y[6] += noise[3*(i*stride+j)+2];
		}
		for(j=0; j<3; j++) output[3*i+j] = y[3*j];
	}

	return_val = output;

"""

headers = ['<math.h>', '<vector>']
arg_names = ['dt', 'N_integrate', 'stride', 'noise', 'y', 'g_syn']


def c_em(Y, dt, N_integrate, stride=1):
	y = Y.flatten()

	noise = sigma*sqrt(dt)/C*randn(3*N_integrate*stride)

	Y_out = inline(code=c_integrate_em,
			arg_names=arg_names,
			headers=headers,
			support_code=function_code,
			extra_compile_args=extra_arch_arg(), extra_link_args=extra_arch_arg(), verbose=0)
	
	Y_out = reshape(Y_out, (N_integrate, 3), 'C')

	return Y_out



def activation(V, V_0, s):
	return 1./(1.+exp(-s*(V-V_0)))

def derivs_one(y):
	[V, h, m] = y
	return [(-g_Na*activation(V, V_Na, s_Na)**3*h*(V-E_Na)-g_K2*m**2*(V-E_K)-g_L*(V-E_L)+I_0)*C_inv, # right-hand side for membrane voltage
			(activation(V, V_h, s_h)-h)*tau_Na_inv, # rhs for Na+ inactivation gating variable
			(activation(V, V_K, s_K)-m)*tau_K2_inv] # rhs for K+ activation gating variable


def derivs_three(y):
	y_1, y_2, y_3 = y[0], y[1], y[2]
	dF = asarray([derivs_one(y_1), derivs_one(y_2), derivs_one(y_3)])

	[activation_1, activation_2, activation_3] =  activation(y[:, 0], theta_syn, s_syn)

	dF[0, 0] -= g_syn*(y[0, 0]-E_syn)*(activation_2+activation_3)*C_inv
	dF[1, 0] -= g_syn*(y[1, 0]-E_syn)*(activation_1+activation_3)*C_inv
	dF[2, 0] -= g_syn*(y[2, 0]-E_syn)*(activation_1+activation_2)*C_inv

	return dF


def em(y, dt, N, stride=1):
	y = asarray(y, dtype=float)
	Y_out = zeros((N, 3), float)
	Y_out[0, :] = y[:, 0]
	sig_sqdt_C = sigma*sqrt(dt)/C

	for n in xrange(1, N):

		for s in xrange(stride):
			y += dt*derivs_three(y)
			y[:, 0] += sig_sqdt_C*randn(3)

		Y_out[n, :] = y[:, 0]
	
	return Y_out


def compute_trace(N=None):

	if N == None:
		N = N_integrate

	X = array([[-40., 0., 0.], [-42., 0., 0.], [-60., 0., 0.]]) # initial condition for the network motif
	if inline_available:
		V = c_em(X, dt/float(stride), N, stride)
	
	else:
		V = em(X, dt/float(stride), N, stride)

	return transpose(V)


V_threshold = -40.
def determine_activity(V):
	blue_active = asarray(V[0] > V_threshold, dtype=float)
	green_active = asarray(V[1] > V_threshold, dtype=float)
	red_active = asarray(V[2] > V_threshold, dtype=float)

	return blue_active, green_active, red_active


vec_bg = array([0., 1.])
vec_gr = array([cos(pi/6.), -sin(pi/6.)])
vec_br = array([-cos(pi/6.), -sin(pi/6.)])
def compute_trajectory(V):
	blue_active, green_active, red_active = determine_activity(V)

	sum_bg = cumsum(blue_active*green_active)
	sum_gr = cumsum(green_active*red_active)
	sum_br = cumsum(blue_active*red_active)

	trajectory = zeros((V.shape[1], 2), float)
	for i in xrange(2):
		trajectory[:, i] = dt*(vec_bg[i]*sum_bg+\
				     vec_gr[i]*sum_gr+\
				     vec_br[i]*sum_br)

	steps = zeros((trajectory.shape[0]-1, 2), float)
	steps[:, 0] = trajectory[1:, 0]-trajectory[:-1, 0]	 # x-trajectory per step
	steps[:, 1] = trajectory[1:, 1]-trajectory[:-1, 1]	 # y-trajectory per step

	length = 0.
	for i in xrange(steps.shape[0]):
		length += sqrt(steps[i, 0]**2 + steps[i, 1]**2)

	return trajectory, length


def get_steps(trajectory):
	increments = zeros((trajectory.shape[0]-1, 2), float)
	increments[:, 0] = trajectory[1:, 0]-trajectory[:-1, 0]	 # x-trajectory per step
	increments[:, 1] = trajectory[1:, 1]-trajectory[:-1, 1]	 # y-trajectory per step

	inc_width = sqrt(increments[:, 0]**2+increments[:, 1]**2)
	
	step_in_progress, step_index, steps = False, [], []
	for i in xrange(increments.shape[0]):
		
		if not inc_width[i] == 0.:

			if not step_in_progress:
				step_index.append(i)
				step_in_progress = True
				steps.append(trajectory[i, :])

		else:
			step_in_progress = False
	
	return step_index, asarray(steps)


def plot_bar(t, activity, value, ax, color='k'):
	i_0 = 0
	da = activity[1:]-activity[:-1]
	for i in xrange(da.size):

		if da[i] > 0.:
			i_0 = i

		if da[i] < 0.:
			ax.plot([t[i_0], t[i]], [value, value], '-', c=color, lw=3.)

	if activity[i] > 0.:
		ax.plot([t[i_0], t[i]], [value, value], '-', c=color, lw=3.)


on_click_axes = {}
def on_click(event):
	try:
		on_click_axes[event.inaxes](event.xdata, event.ydata)
	
	except:
		pass


def set_params(new_g, new_sigma2):
	global g_syn, sigma
	tx_0.set_text('')


	if g_syn < 0. or new_sigma2 < 0.:
		print 'Invalid value of g_syn or \sigma^2.'
		return

	g_syn = new_g/1000.
	sigma = sqrt(new_sigma2)

	print 'setting g_syn =', g_syn*1000., 'pS'
	print 'setting sigma^2 =', sigma**2, 'pA^2/s'

	p_dot.set_data([1000.*g_syn], [sigma**2])
	draw()


def new_trajectory(x, y):

	tx.set_text('Computing a\nnew trajectory')
	draw()

	V = compute_trace(N=N_integrate)
	
	li_blue.set_ydata(V[0])
	li_green.set_ydata(V[1]-60.)
	li_red.set_ydata(V[2]-120.)


	b_activity, g_activity, r_activity = determine_activity(V)

	for i in xrange(len(ax_trace.collections)):
		ax_trace.collections.pop()

	for i in xrange(len(ax_trace.lines)-7):
		ax_trace.lines.pop()

	plot_bar(t, (b_activity*g_activity), A_level, ax_trace, color=color[0])
	plot_bar(t, (g_activity*r_activity), B_level, ax_trace, color=color[1])
	plot_bar(t, (b_activity*r_activity), C_level, ax_trace, color=color[2])

	coll = collections.BrokenBarHCollection.span_where(t, ymin=A_level+5., ymax=40., where=(b_activity*g_activity), alpha=0.3, facecolor=color[0], edgecolor='')
	ax_trace.add_collection(coll)
	coll = collections.BrokenBarHCollection.span_where(t, ymin=B_level+5., ymax=40., where=(g_activity*r_activity), alpha=0.3, facecolor=color[1], edgecolor='')
	ax_trace.add_collection(coll)
	coll = collections.BrokenBarHCollection.span_where(t, ymin=C_level+5., ymax=40., where=(b_activity*r_activity), alpha=0.3, facecolor=color[2], edgecolor='')
	ax_trace.add_collection(coll)


	trajectory, length = compute_trajectory(V)
	n_step, steps = get_steps(trajectory)

	li_traj.set_data(trajectory[:, 0], trajectory[:, 1])
	li_steps.set_data(steps[:, 0], steps[:, 1])
	scale_ax(trajectory[:, 0], trajectory[:, 1], ax_space)

	annot.xy = (trajectory[n_step[1], 0], trajectory[n_step[1], 1])
	annot.xytext = (trajectory[n_step[0], 0], trajectory[n_step[0], 1])

	tx.set_text('')
	draw()


def scale_ax(x, y, ax):
	ax.set_xlim(x.min()-5., x.max()+5.)
	ax.set_ylim(y.min()-5., y.max()+5.)

	



#===


sigma = 0.06 # pA/sqrt(s)
g_syn = 0.017 # nS

dt = 0.03
stride = 30
T = 100.


#===		load initial conditions

N_integrate = int(T/dt)
t = dt*arange(N_integrate)


#"""

#import time
#inline_available = False
#t_0 = time.time()
V = compute_trace(N=N_integrate)
#print 'elapsed time:', time.time()-t_0


#exit(0)



b_activity, g_activity, r_activity = determine_activity(V)
trajectory, length = compute_trajectory(V)
n_step, steps = get_steps(trajectory)
#"""

#===		solve all initial conditions

fontsize = 20
left, bottom = 0.13, 0.06

space_box = [left, bottom, 0.95-left, 0.6]
space_box_inset = [0.11+left, 0.01+bottom, 0.33, 0.28]
params_box = [0.12, 0.06+space_box[1]+space_box[3], 0.28, 0.25]
diff_box = [0.09+space_box[0]+space_box[2], bottom, 0., space_box[3]]
diff_box[2] = 0.97-diff_box[0]
trace_box = [0.1+params_box[0]+params_box[2], 0.03+bottom+space_box[3], 0., 0.]
trace_box[2] = 0.97-trace_box[0]
trace_box[3] = 0.97-trace_box[1]
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

color = ['m', '#777700', '#000000']





fig = plt.figure(figsize=(9, 11))




ax_params = axes(params_box)
on_click_axes[ax_params] = set_params
p_dot, = plot([1000.*g_syn], [sigma**2], 'ro')
tx_0 = text(7., 0.001, 'CLICK HERE!', fontsize=14, bbox=bbox_props)
xlim(0., 20.)
ylim(0., 0.01)
grid()

xlabel(r'$g_{\rm inh}$ (pS)', fontsize=16)
ylabel(r'$\sigma^2$ (pA$^2$/s)', fontsize=16)



ax_trace = axes(trace_box, frameon=False, xticks=[], yticks=[])

li_blue, = plot(t, V[0], 'b-', lw=0.5)
li_green, = plot(t, V[1]-60., 'g-', lw=0.5)
li_red, = plot(t, V[2]-120., 'r-', lw=0.5)

plot(array([t[-1]-10., t[-1], t[-1]])+1., array([-180., -180., -130])-8., 'k-', lw=3.)
text(t[-1]-9., -212, '10sec.', fontsize=fontsize-4)
text(t[-1]+1.7, -132, '50mV', fontsize=fontsize-4, rotation=90)

A_level, B_level, C_level = -210., -240., -270.

c = '#888888'
axhline(y=A_level-5., color=c, ls=':')
axhline(y=B_level-5., color=c, ls=':')
axhline(y=C_level-5., color=c, ls=':')

text(-3., A_level-5., 'A', fontsize=fontsize, color=color[0])
text(-3., B_level-5., 'B', fontsize=fontsize, color=color[1])
text(-3., C_level-5., 'C', fontsize=fontsize, color=color[2])
plot_bar(t, (b_activity*g_activity), A_level, ax_trace, color=color[0])
plot_bar(t, (g_activity*r_activity), B_level, ax_trace, color=color[1])
plot_bar(t, (b_activity*r_activity), C_level, ax_trace, color=color[2])

coll = collections.BrokenBarHCollection.span_where(t, ymin=A_level+5., ymax=40., where=(b_activity*g_activity), alpha=0.3, facecolor=color[0], edgecolor='')
ax_trace.add_collection(coll)
coll = collections.BrokenBarHCollection.span_where(t, ymin=B_level+5., ymax=40., where=(g_activity*r_activity), alpha=0.3, facecolor=color[1], edgecolor='')
ax_trace.add_collection(coll)
coll = collections.BrokenBarHCollection.span_where(t, ymin=C_level+5., ymax=40., where=(b_activity*r_activity), alpha=0.3, facecolor=color[2], edgecolor='')
ax_trace.add_collection(coll)

ylim(-280., 50.)
xlim(0., t[-1]+2.)





ax_space = axes(space_box)
on_click_axes[ax_space] = new_trajectory
li_traj, = plot(trajectory[:, 0], trajectory[:, 1], '-', c='#555555', lw=2.0)
li_steps, = plot(steps[:, 0], steps[:, 1], 'co', ms=1.0)

scale_ax(trajectory[:, 0], trajectory[:, 1], ax_space)
grid()
xmin, xmax = xlim()
ymin, ymax = ylim()
lim_x, lim_y = xmax-xmin, ymax-ymin

annot = annotate("",
    xy=(trajectory[n_step[1], 0], trajectory[n_step[1], 1]), xycoords='data',
    xytext=(trajectory[n_step[0], 0], trajectory[n_step[0], 1]), textcoords='data',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=1.9))

tx = text(-3., 0., 'CLICK HERE!', fontsize=14, bbox=bbox_props)

xlabel(r'x-direction (m)', fontsize=fontsize)
ylabel(r'y-direction (m)', fontsize=fontsize)



ax_dir = axes([space_box[0]+0.01, space_box[1]+space_box[3]-0.20, 0.20, 0.18], xticks=[], yticks=[], frameon=False)

X_r, X_g, X_b = vec_bg, vec_br, vec_gr
plot([0.], [0.], 'co', ms=15)
text(X_r[0]-0.12, X_r[1]-0.1, 'A', color=color[0], fontsize=fontsize)
text(X_b[0]-0.07, X_b[1], 'B', color=color[1], fontsize=fontsize)
text(X_g[0]-0.11, X_g[1], 'C', color=color[2], fontsize=fontsize)
annotate('', (X_r[0], X_r[1]), (0., 0.), arrowprops={'lw':0.5, 'color':color[0], 'shrink':0.2})
annotate('', (X_g[0], X_g[1]), (0., 0.), arrowprops={'lw':0.5, 'color':color[2], 'shrink':0.2})
annotate('', (X_b[0], X_b[1]), (0., 0.), arrowprops={'lw':0.5, 'color':color[1], 'shrink':0.2})

xlim(-1.1, 1.1)
ylim(-0.7, 1.3)

cid = fig.canvas.mpl_connect('button_press_event', on_click)

show()













